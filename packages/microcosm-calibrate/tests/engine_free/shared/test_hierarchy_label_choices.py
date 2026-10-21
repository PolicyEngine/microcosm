"""Labels chosen for identifiers that a country's sources spell more than one way.

Diagnostics schema 9 holds one label per identifier, so the writer refuses
rows that label one identifier two ways unless a label is chosen for it. The
choices below cover the spellings found in published releases: UK publishers'
names for one area (HMRC's "Hartlepool UA" beside ONS's "Hartlepool"), which
reach the rows as values of Chronicle's ``geography`` dimension, and three US
dimension values that two IRS SOI tables or two Census tables word
differently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyDimension,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
    score_targets,
    write_calibration_diagnostics,
)
from microcosm.calibrate.geography_constants import UK_GEOGRAPHY_ID_TO_LABEL
from microcosm.calibrate.hierarchy_labels import (
    US_DIMENSION_VALUE_LABEL_CHOICES,
    hierarchy_label_choices,
)
from microcosm.diagnostics import HierarchyLabelChoices

AGI = "us:statutes/26/62#adjusted_gross_income"


def test_uk_choices_name_areas_from_the_authoritative_catalog() -> None:
    choices = hierarchy_label_choices("uk")

    assert dict(choices.geographies) == dict(UK_GEOGRAPHY_ID_TO_LABEL)
    assert choices.geography_dimensions == frozenset({"geography"})
    assert dict(choices.dimension_values) == {}
    assert hierarchy_label_choices("UK") == choices


def test_us_choices_name_the_three_values_us_sources_word_differently() -> None:
    choices = hierarchy_label_choices("us")

    # The labels chosen in review on 2026-10-21, which the US feed re-pin also
    # carries from Chronicle; older feeds and releases keep both spellings.
    assert dict(choices.dimension_values) == {
        (AGI, "under_1"): "No adjusted gross income",
        (AGI, "all"): "All returns, total",
        ("age", "85_plus"): "Age 85 plus",
    }
    assert dict(US_DIMENSION_VALUE_LABEL_CHOICES) == dict(choices.dimension_values)
    # Each choice covers exactly the spellings found in release 0fb05b6 and
    # registered by chronicle#377; a third spelling is still refused.
    assert dict(choices.dimension_value_spellings) == {
        (AGI, "under_1"): frozenset(
            {"No adjusted gross income", "No adjusted gross income and deficit"}
        ),
        (AGI, "all"): frozenset({"All returns", "All returns, total"}),
        ("age", "85_plus"): frozenset({"Age 85 plus", "Age 85 and over"}),
    }
    assert dict(choices.geographies) == {}
    assert choices.geography_dimensions == frozenset()


@pytest.mark.parametrize("country", ["be", "nz", "toy", None])
def test_other_countries_choose_no_labels(country: str | None) -> None:
    assert hierarchy_label_choices(country) == HierarchyLabelChoices()


def _spec(
    name: str,
    value: float,
    *,
    geography: HierarchyGeography,
    dimension: HierarchyDimension,
) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="household",
        measure="household_count",
        value=value,
        period=2025,
        source="fixture | Fixture table | https://example.test/fixture",
        family="fixture",
        metadata={},
        hierarchy=CalibrationHierarchy(
            provider=HierarchyNode("fixture", "Fixture publisher"),
            category=HierarchyCategory("fixture.households", "Households", "fixture"),
            geography=geography,
            dimensions=(dimension,),
            target=HierarchyNode(name, name.replace("_", " ")),
        ),
    )


def _uk_area_registry(country: str, population: float) -> TargetRegistry:
    hartlepool = HierarchyGeography("E06000001", "Hartlepool", level="local_authority")
    return TargetRegistry(
        (
            _spec(
                "hmrc_households_e06000001",
                population,
                geography=hartlepool,
                dimension=HierarchyDimension(
                    "geography", "Geography", "e06000001", "Hartlepool UA"
                ),
            ),
            _spec(
                "ons_households_e06000001",
                population,
                geography=hartlepool,
                dimension=HierarchyDimension(
                    "geography", "Geography", "e06000001", "Hartlepool"
                ),
            ),
        ),
        country=country,
    )


def _us_band_registry(country: str, population: float) -> TargetRegistry:
    nation = HierarchyGeography("0100000US", "United States", level="country")
    return TargetRegistry(
        (
            _spec(
                "soi_returns_under_1",
                population,
                geography=nation,
                dimension=HierarchyDimension(
                    AGI, "Adjusted gross income", "under_1", "No adjusted gross income"
                ),
            ),
            _spec(
                "eitc_returns_under_1",
                population,
                geography=nation,
                dimension=HierarchyDimension(
                    AGI,
                    "Adjusted gross income",
                    "under_1",
                    "No adjusted gross income and deficit",
                ),
            ),
        ),
        country=country,
    )


def _written_values(path: Path) -> list[dict]:
    (dimension,) = json.loads(path.read_text())["hierarchy_catalogs"]["dimensions"]
    return dimension["values"]


def test_the_writer_applies_the_registry_country_choices(
    feasible_frame,
    tmp_path: Path,
) -> None:
    frame, truths = feasible_frame()
    registry = _us_band_registry("us", truths["population"])
    result = score_targets(frame, registry.to_target_set())
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_calibration_diagnostics(result, path, target_registry=registry)

    assert outcome.status == "available"
    assert _written_values(path) == [
        {"id": "under_1", "label": "No adjusted gross income"}
    ]


def test_the_uk_choices_give_an_area_its_own_geography_label(
    feasible_frame,
    tmp_path: Path,
) -> None:
    # A UK registry needs the UK extension block, which the UK build's writer
    # adds (microcosm.build.uk_runtime.diagnostics); the generic writer is
    # given the UK choices explicitly here.
    frame, truths = feasible_frame()
    registry = _uk_area_registry("be", truths["population"])
    result = score_targets(frame, registry.to_target_set())
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_calibration_diagnostics(
        result,
        path,
        target_registry=registry,
        label_choices=hierarchy_label_choices("uk"),
    )

    assert outcome.status == "available"
    assert _written_values(path) == [{"id": "e06000001", "label": "Hartlepool"}]


@pytest.mark.parametrize("registry_for", [_uk_area_registry, _us_band_registry])
def test_a_country_without_choices_is_refused_the_same_spellings(
    feasible_frame,
    tmp_path: Path,
    registry_for,
) -> None:
    frame, truths = feasible_frame()
    registry = registry_for("be", truths["population"])
    result = score_targets(frame, registry.to_target_set())
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_calibration_diagnostics(result, path, target_registry=registry)

    assert outcome.status == "failed"
    assert outcome.error_code == "label_conflict"
    assert not path.exists()
