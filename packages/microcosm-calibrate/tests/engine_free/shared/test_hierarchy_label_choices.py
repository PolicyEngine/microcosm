"""Labels chosen for identifiers that a country's sources spell more than one way.

Diagnostics schema 9 holds one label per identifier, so the writer refuses
rows that label one identifier two ways unless a label is chosen for it. The
choices below cover the spellings found in published releases: UK publishers'
names for one area (HMRC's "Hartlepool UA" beside ONS's "Hartlepool"), which
reach the rows as values of Chronicle's ``geography`` dimension, and three US
dimension values that two IRS SOI tables or two Census tables word
differently. They also cover the Belgian ``measure`` = ``country_total``
value, which each Belgian Chronicle package labels its own way.
"""

from __future__ import annotations

import json
import logging
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
COUNTRY_TOTAL = ("measure", "country_total")

#: Belgium's country-total targets (``be/target_references.json``), each read
#: from its own Chronicle package, which labels the group-by value ``measure``
#: = ``country_total`` its own way (``source_package.yaml`` at Chronicle
#: ``6d74e42``, registered there as reviewed variants). The NBB target has
#: ``target_role`` "validation"; the other three are calibration targets.
BELGIAN_COUNTRY_TOTAL_TARGETS = (
    (
        "spf_finances_pit_total",
        HierarchyNode("spf_finances", "FPS Finance"),
        HierarchyCategory(
            "spf_finances.income_tax", "Personal income tax", "spf_finances"
        ),
        "Belgium PIT federal and local tax before withholding",
    ),
    (
        "onss_employee_contribution_total",
        HierarchyNode("onss", "National Social Security Office"),
        HierarchyCategory(
            "onss.social_security", "Social security contributions", "onss"
        ),
        "Worker-borne personal social security contributions",
    ),
    (
        "onem_unemployment_caseload",
        HierarchyNode("onem", "National Employment Office"),
        HierarchyCategory("onem.caseloads", "Benefit caseloads", "onem"),
        "Complete unemployment benefit recipients",
    ),
    (
        "nbb_household_disposable_income",
        HierarchyNode("nbb", "National Bank of Belgium"),
        HierarchyCategory("nbb.national_accounts", "National accounts", "nbb"),
        "Household disposable income gross",
    ),
)
BELGIAN_COUNTRY_TOTAL_SPELLINGS = frozenset(
    value_label for *_target, value_label in BELGIAN_COUNTRY_TOTAL_TARGETS
)


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


def test_the_belgian_choice_covers_the_four_country_total_spellings() -> None:
    choices = hierarchy_label_choices("be")

    # The label chosen in review on 2026-10-22, which is also the label
    # Chronicle's canonical register gives this value (chronicle#377).
    assert dict(choices.dimension_values) == {COUNTRY_TOTAL: "Country total"}
    # The choice covers exactly the four package spellings Chronicle registers
    # as reviewed variants; another spelling is still refused.
    assert dict(choices.dimension_value_spellings) == {
        COUNTRY_TOTAL: BELGIAN_COUNTRY_TOTAL_SPELLINGS
    }
    assert len(BELGIAN_COUNTRY_TOTAL_SPELLINGS) == 4
    assert dict(choices.geographies) == {}
    assert choices.geography_dimensions == frozenset()
    assert hierarchy_label_choices("BE") == choices


@pytest.mark.parametrize("country", ["nz", "toy", None])
def test_other_countries_choose_no_labels(country: str | None) -> None:
    assert hierarchy_label_choices(country) == HierarchyLabelChoices()


def _spec(
    name: str,
    value: float,
    *,
    geography: HierarchyGeography,
    dimension: HierarchyDimension,
    provider: HierarchyNode | None = None,
    category: HierarchyCategory | None = None,
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
            provider=provider or HierarchyNode("fixture", "Fixture publisher"),
            category=category
            or HierarchyCategory("fixture.households", "Households", "fixture"),
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


def _belgian_country_total_registry(
    population: float,
    *,
    extra_value_label: str | None = None,
) -> TargetRegistry:
    """Belgium's country-total targets, each with its package's value label.

    ``extra_value_label`` adds a fifth target from another package whose
    spelling of the value nobody has reviewed.
    """

    belgium = HierarchyGeography("BE", "Belgium", level="country")
    targets = list(BELGIAN_COUNTRY_TOTAL_TARGETS)
    if extra_value_label is not None:
        targets.append(
            (
                "statbel_country_total",
                HierarchyNode("statbel", "Statbel"),
                HierarchyCategory("statbel.demography", "Demography", "statbel"),
                extra_value_label,
            )
        )
    return TargetRegistry(
        tuple(
            _spec(
                target_name,
                population,
                geography=belgium,
                dimension=HierarchyDimension(
                    "measure", "Measure", "country_total", value_label
                ),
                provider=provider,
                category=category,
            )
            for target_name, provider, category, value_label in targets
        ),
        country="be",
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


def test_a_belgian_file_is_written_with_country_total_and_a_warning(
    feasible_frame,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Each Belgian package labels measure = country_total its own way. The
    # transport graph's diagnostics step takes the same choices (by the
    # registry's country) and raises on a refusal, so without a Belgian
    # choice a Belgian build fails.
    frame, truths = feasible_frame()
    registry = _belgian_country_total_registry(truths["population"])
    result = score_targets(frame, registry.to_target_set())
    path = tmp_path / "calibration_diagnostics.json"

    with caplog.at_level("WARNING", logger="microcosm.diagnostics.catalogs"):
        outcome = write_calibration_diagnostics(result, path, target_registry=registry)

    assert outcome.status == "available"
    written = json.loads(path.read_text())
    assert written["schema_version"] == 9
    assert _written_values(path) == [{"id": "country_total", "label": "Country total"}]
    (record,) = [
        record
        for record in caplog.records
        if record.name == "microcosm.diagnostics.catalogs"
    ]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "label 1 identifier(s) more than one way" in message
    assert "dimension measure value country_total: " in message
    for value_label in BELGIAN_COUNTRY_TOTAL_SPELLINGS:
        assert repr(value_label) in message
    assert message.endswith("-> 'Country total'")


def test_a_belgian_spelling_nobody_reviewed_is_still_refused(
    feasible_frame,
    tmp_path: Path,
) -> None:
    frame, truths = feasible_frame()
    registry = _belgian_country_total_registry(
        truths["population"], extra_value_label="Belgium country total"
    )
    result = score_targets(frame, registry.to_target_set())
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_calibration_diagnostics(result, path, target_registry=registry)

    assert outcome.status == "failed"
    assert outcome.error_code == "label_conflict"
    assert (
        "dimension measure value country_total (a spelling its reviewed list "
        "does not include)"
    ) in outcome.message
    assert not path.exists()


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
    registry = registry_for("nz", truths["population"])
    result = score_targets(frame, registry.to_target_set())
    path = tmp_path / "calibration_diagnostics.json"

    outcome = write_calibration_diagnostics(result, path, target_registry=registry)

    assert outcome.status == "failed"
    assert outcome.error_code == "label_conflict"
    assert not path.exists()
