"""The three release files a spec country's build adds (tracker 5b · 13).

BE's release contract requires ``source_coverage.json``, ``demographics.json``
and ``reform_validation.json`` beside the three base files. These tests hold
their contents to what BE's spec declares and what the build's calibration
diagnostics record, on fixture diagnostics; no build runs.
"""

from __future__ import annotations

import pytest

from microcosm.build.spec_release_files import (
    SPEC_RELEASE_FILES,
    demographics_payload,
    reform_validation_payload,
    source_coverage_payload,
)
from microcosm.build.transport.compose import load_transport_spec
from microcosm.data import contract as contract_module
from test_support.microcosm_build.spec_release import (
    calibration_diagnostics,
    demography_estimate,
    demography_references,
    demography_target,
)

#: The fields of ``demographics.json``: the US file's, plus the country, the
#: cells it sums and whether they were calibrated.
DEMOGRAPHICS_FIELDS = {
    "schema_version",
    "country",
    "period",
    "measure",
    "in_sample",
    "total_population",
    "benchmark_total_population",
    "benchmark_source",
    "age_bands",
    "cells",
}
#: The fields of ``source_coverage.json``: the US file's, plus the country.
SOURCE_COVERAGE_FIELDS = {
    "schema_version",
    "classification",
    "country",
    "source_contract",
    "gate",
    "coverage_summary",
    "hard_target_families",
    "validation_only_families",
    "source_gap_families",
    "active_target_aliases",
    "active_target_families",
    "missing_hard_targets",
    "reviewed_exclusions",
    "validation_only_activated",
}


@pytest.fixture(scope="module")
def be_spec() -> dict:
    return load_transport_spec("be")


def _source_coverage(be_spec: dict, diagnostics: dict) -> dict:
    resources = be_spec["resources"]
    return source_coverage_payload(
        country="be",
        target_references=resources["target_references"],
        holdout_references=resources["holdout_references"],
        diagnostics=diagnostics,
        source_contract={"chronicle_commit": "a" * 40},
    )


def test_demographics_reports_each_demography_cell_from_the_calibration_rows(
    be_spec: dict,
) -> None:
    references = demography_references("be")
    payload = demographics_payload(
        country="be",
        target_references=be_spec["resources"]["target_references"],
        diagnostics=calibration_diagnostics("be"),
        period=2026,
    )

    assert DEMOGRAPHICS_FIELDS <= set(payload)
    assert payload["schema_version"] == 1
    assert (payload["country"], payload["period"]) == ("be", 2026)
    assert payload["measure"] == "person_weight"
    # BE calibrates its demography, so the cells' fit is in-sample.
    assert payload["in_sample"] is True
    assert "Statbel" in payload["benchmark_source"]
    cells = payload["cells"]
    assert [cell["target"] for cell in cells] == [ref["name"] for ref in references]
    first, selector = cells[0], references[0]["ledger_selector"]
    assert first["geography_level"] == selector["geography_level"]
    assert first["geography_id"] == selector["geography_id"]
    assert first["dimensions"] == selector["dimension_values"]
    assert first["population"] == pytest.approx(demography_estimate(0))
    assert first["benchmark"] == pytest.approx(demography_target(0))
    assert first["relative_error"] == pytest.approx(0.01)
    populations = [demography_estimate(index) for index in range(len(references))]
    benchmarks = [demography_target(index) for index in range(len(references))]
    assert payload["total_population"] == pytest.approx(sum(populations))
    assert payload["benchmark_total_population"] == pytest.approx(sum(benchmarks))


def test_demographics_sums_the_cells_into_age_bands_in_spec_order(
    be_spec: dict,
) -> None:
    references = demography_references("be")
    payload = demographics_payload(
        country="be",
        target_references=be_spec["resources"]["target_references"],
        diagnostics=calibration_diagnostics("be"),
        period=2026,
    )

    assert DEMOGRAPHICS_FIELDS <= set(payload)
    bands: dict[str, list[int]] = {}
    for index, reference in enumerate(references):
        band = reference["ledger_selector"]["dimension_values"]["person.age_band"]
        bands.setdefault(band, []).append(index)
    rows = payload["age_bands"]
    assert [row["label"] for row in rows] == list(bands)
    for row in rows:
        indexes = bands[row["label"]]
        assert row["population"] == pytest.approx(
            sum(demography_estimate(index) for index in indexes)
        )
        assert row["benchmark"] == pytest.approx(
            sum(demography_target(index) for index in indexes)
        )
    assert sum(row["share"] for row in rows) == pytest.approx(1.0)
    assert sum(row["benchmark_share"] for row in rows) == pytest.approx(1.0)


def test_demographics_leaves_a_cell_the_build_did_not_calibrate_unmeasured(
    be_spec: dict,
) -> None:
    omitted = demography_references("be")[0]["name"]
    payload = demographics_payload(
        country="be",
        target_references=be_spec["resources"]["target_references"],
        diagnostics=calibration_diagnostics("be", omit=[omitted]),
        period=2026,
    )

    assert DEMOGRAPHICS_FIELDS <= set(payload)
    (cell,) = [cell for cell in payload["cells"] if cell["target"] == omitted]
    assert (cell["population"], cell["benchmark"], cell["relative_error"]) == (
        None,
        None,
        None,
    )
    # Without every cell the totals are unknown, not a partial sum.
    assert payload["total_population"] is None
    assert payload["benchmark_total_population"] is None


def test_source_coverage_lists_the_spec_families_by_role(be_spec: dict) -> None:
    resources = be_spec["resources"]
    payload = _source_coverage(be_spec, calibration_diagnostics("be"))

    assert SOURCE_COVERAGE_FIELDS <= set(payload)
    assert payload["schema_version"] == 1
    assert payload["classification"] == "release_gate"
    assert payload["country"] == "be"
    assert payload["source_contract"]["name"] == "be_source_coverage"
    assert payload["source_contract"]["chronicle_commit"] == "a" * 40
    assert payload["gate"] == {
        "name": "be_source_coverage",
        "passed": True,
        "failures": [],
    }
    # Active calibration families are hard targets; families whose
    # references are all parked are source gaps; held-out families are
    # validation-only.
    assert set(payload["hard_target_families"]) == {"statbel.demography"}
    target_categories = set(resources["target_references"]["hierarchy"]["categories"])
    assert set(payload["source_gap_families"]) == target_categories - {
        "statbel.demography"
    }
    assert set(payload["validation_only_families"]) == set(
        resources["holdout_references"]["hierarchy"]["categories"]
    )
    demography = payload["hard_target_families"]["statbel.demography"]
    assert set(demography["package_aliases"]) == {
        reference["name"] for reference in demography_references("be")
    }
    assert payload["missing_hard_targets"] == []


def test_source_coverage_fails_when_the_build_left_out_an_active_reference(
    be_spec: dict,
) -> None:
    omitted = demography_references("be")[3]["name"]
    payload = _source_coverage(be_spec, calibration_diagnostics("be", omit=[omitted]))

    assert SOURCE_COVERAGE_FIELDS <= set(payload)
    assert payload["gate"]["passed"] is False
    assert payload["missing_hard_targets"] == [omitted]


def test_source_coverage_fails_when_the_build_calibrated_a_held_out_reference(
    be_spec: dict,
) -> None:
    holdout = be_spec["resources"]["holdout_references"]["target_references"][0]
    diagnostics = calibration_diagnostics("be")
    row = dict(diagnostics["targets"][0])
    row["name"], row["target_name"] = f"{holdout['name']}@2026", holdout["name"]
    diagnostics["targets"].append(row)

    payload = _source_coverage(be_spec, diagnostics)

    assert SOURCE_COVERAGE_FIELDS <= set(payload)
    assert payload["gate"]["passed"] is False
    assert holdout["category_id"] in payload["validation_only_activated"]


def test_the_added_files_are_the_ones_the_be_contract_declares() -> None:
    # The data shard names BE's added files by artifact key; the build
    # shard's writers must write the same files under the same keys.
    assert dict(SPEC_RELEASE_FILES) == contract_module._BE_ADDED_RELEASE_ARTIFACTS


def test_reform_validation_carries_no_reforms_yet() -> None:
    # No scored BE reform exists until microcosm#264; nothing was simulated.
    assert reform_validation_payload(country="be", period=2026) == {
        "schema_version": 1,
        "country": "be",
        "baseline_period": 2026,
        "scoring_window": None,
        "out_of_sample_simulated": False,
        "reforms": [],
    }
