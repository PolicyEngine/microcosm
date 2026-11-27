"""The release files a spec country's build adds to the base three.

A spec country's release contract (``microcosm/build/<country>/
release_contract.json``) can require ``source_coverage.json``,
``demographics.json`` and ``reform_validation.json`` beside
``build_manifest.json``, ``release_manifest.json`` and
``calibration_diagnostics.json``, as BE's does (tracker 5b · 13). This module
builds each file's payload from what the country's spec declares and what the
build's calibration diagnostics record; it reads no population and runs no
simulation. :mod:`microcosm.build.spec_release_assembly` writes them into a
release directory.

- ``source_coverage.json`` has the US file's fields
  (:mod:`microcosm.build.us_runtime.source_coverage`), from the shared
  :func:`microcosm.build.gates.source_coverage_gate`. Its source families are
  the spec's target categories, or each reference's ``family`` where the spec
  declares no category for it (as NZ's does): a family with an executable
  calibration reference is a hard target, one whose references are all parked
  is a source gap, and each held-out family is validation-only. The gate fails when the
  build did not calibrate an executable reference, or calibrated a held-out
  one.
- ``demographics.json`` has the US file's fields
  (:mod:`microcosm.build.us_runtime.demographics`), except each age band's
  ``min_age`` and ``max_age``, which the spec's age-band values do not give,
  for the spec's executable demography references: each one's calibrated population and its benchmark,
  from the build's calibration rows, and their sums by age band. A country
  that calibrates its demography, as BE does, publishes in-sample fit here,
  where the US publishes an uncalibrated age distribution.
- ``reform_validation.json`` has the US file's schema 1 fields with no
  reforms until the country has scored ones (microcosm#264 for BE), and
  ``out_of_sample_simulated`` true, as the US writer records it when there is
  no out-of-sample reform.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from microcosm.build.gates import source_coverage_gate

__all__ = [
    "AGE_BAND_DIMENSION",
    "DEMOGRAPHICS_SCHEMA_VERSION",
    "DEMOGRAPHY_FAMILY",
    "REFORM_VALIDATION_SCHEMA_VERSION",
    "SOURCE_COVERAGE_SCHEMA_VERSION",
    "SPEC_RELEASE_FILES",
    "demographics_payload",
    "reform_validation_payload",
    "source_coverage_name",
    "source_coverage_payload",
]

#: Each added release file by the artifact key a build declares it under,
#: which is also its role in a build folder (``_BE_ADDED_RELEASE_ARTIFACTS``
#: in :mod:`microcosm.data.contract`).
SPEC_RELEASE_FILES: Mapping[str, str] = {
    "source_coverage": "source_coverage.json",
    "demographics": "demographics.json",
    "reform_validation": "reform_validation.json",
}
SOURCE_COVERAGE_SCHEMA_VERSION = 1
DEMOGRAPHICS_SCHEMA_VERSION = 1
REFORM_VALIDATION_SCHEMA_VERSION = 1
#: The target-reference family whose references are population counts.
DEMOGRAPHY_FAMILY = "demography"
#: The Ledger dimension that names a demography reference's age band.
AGE_BAND_DIMENSION = "person.age_band"
#: Activation statuses the target compiler treats as executable
#: (``ledger_targets._require_executable_reference``): absent, empty or
#: ``"active"``.
_EXECUTABLE_STATUSES = frozenset({"", "active"})


def _references(document: Mapping | None) -> tuple[Mapping, ...]:
    if document is None:
        return ()
    rows = document.get("target_references", ())
    return tuple(row for row in rows if isinstance(row, Mapping))


def _categories(document: Mapping | None) -> Mapping[str, Mapping]:
    if document is None:
        return {}
    hierarchy = document.get("hierarchy", {})
    categories = (
        hierarchy.get("categories", {}) if isinstance(hierarchy, Mapping) else {}
    )
    return categories if isinstance(categories, Mapping) else {}


def _activation_status(reference: Mapping) -> str:
    metadata = reference.get("metadata", {})
    status = (
        metadata.get("activation_status", "") if isinstance(metadata, Mapping) else ""
    )
    return status if isinstance(status, str) else ""


def _executable(reference: Mapping) -> bool:
    return _activation_status(reference) in _EXECUTABLE_STATUSES


def _target_name(row: Mapping) -> str | None:
    """The reference a calibration row measures: its ``target_name``, or its
    ``name`` without the ``@<period>`` suffix."""
    name = row.get("target_name")
    if isinstance(name, str) and name:
        return name
    name = row.get("name")
    if isinstance(name, str) and name:
        return name.partition("@")[0]
    return None


def _calibration_rows(diagnostics: Mapping) -> dict[str, Mapping]:
    """The build's calibration rows by the reference each measures."""
    rows: dict[str, Mapping] = {}
    for row in diagnostics.get("targets", ()):
        if not isinstance(row, Mapping):
            continue
        name = _target_name(row)
        if name is not None:
            rows.setdefault(name, row)
    return rows


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _json_ready(value: object) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def source_coverage_name(country: str) -> str:
    """The source coverage contract and gate name, as ``us_source_coverage``."""
    return f"{country}_source_coverage"


def _family_of(reference: Mapping, categories: Mapping[str, Mapping]) -> str:
    """The source family a reference counts toward: its declared category,
    else its ``family`` (a spec that declares no categories, as NZ's)."""
    category = reference.get("category_id")
    if isinstance(category, str) and category in categories:
        return category
    family = reference.get("family")
    return family if isinstance(family, str) and family else "uncategorised"


def _families(document: Mapping | None) -> dict[str, tuple[str, list[Mapping]]]:
    """Each source family of a reference document: its label and references,
    in the order the document first names each."""
    categories = _categories(document)
    families: dict[str, tuple[str, list[Mapping]]] = {}
    for reference in _references(document):
        family = _family_of(reference, categories)
        category = categories.get(family, {})
        label = (
            category.get("label", family) if isinstance(category, Mapping) else family
        )
        families.setdefault(family, (str(label), []))[1].append(reference)
    for family, category in categories.items():
        label = (
            category.get("label", family) if isinstance(category, Mapping) else family
        )
        families.setdefault(family, (str(label), []))
    return families


#: Appended to a held-out family's ID when a target family has the same one,
#: so the two stay apart in the source coverage check.
_HELD_OUT_SUFFIX = ".held_out"


def _source_families(
    target_references: Mapping, holdout_references: Mapping | None
) -> tuple[dict[str, tuple[str, list[Mapping]]], dict[str, tuple[str, list[Mapping]]]]:
    """The target and held-out families, with no ID in both.

    A held-out family whose ID a target family also has (possible where the
    spec declares no categories and both documents name one ``family``) takes
    :data:`_HELD_OUT_SUFFIX`, so a calibrated target never counts as a
    calibrated held-out reference.
    """
    targets = _families(target_references)
    holdouts = {
        (family + _HELD_OUT_SUFFIX if family in targets else family): entry
        for family, entry in _families(holdout_references).items()
    }
    return targets, holdouts


def _coverage_entries(
    targets: Mapping[str, tuple[str, list[Mapping]]],
    holdouts: Mapping[str, tuple[str, list[Mapping]]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for family, (label, members) in targets.items():
        executable = tuple(str(ref["name"]) for ref in members if _executable(ref))
        if executable:
            entries.append(
                {
                    "family_id": family,
                    "label": label,
                    "role": "hard_target",
                    "package_aliases": executable,
                }
            )
        else:
            entries.append(
                {
                    "family_id": family,
                    "label": label,
                    "role": "source_gap",
                    "missing_source_packages": tuple(
                        f"{ref['name']}: {_activation_status(ref) or 'unset'}"
                        for ref in members
                    ),
                }
            )
    for family, (label, members) in holdouts.items():
        entries.append(
            {
                "family_id": family,
                "label": label,
                "role": "validation_only",
                "package_aliases": tuple(str(ref["name"]) for ref in members),
            }
        )
    return entries


def source_coverage_payload(
    *,
    country: str,
    target_references: Mapping,
    holdout_references: Mapping | None,
    diagnostics: Mapping,
    source_contract: Mapping[str, object],
) -> dict[str, object]:
    """``source_coverage.json``: the spec's source families against the build.

    Args:
        country: The spec's country code (``"be"``).
        target_references: The spec's ``target_references.json``.
        holdout_references: The spec's ``holdout_references.json``, if any.
        diagnostics: The build's calibration diagnostics; its rows are the
            references the build calibrated.
        source_contract: What the families were read from, such as the
            Chronicle pin and the resources' digests; the payload adds the
            contract's name and country.
    """
    targets, holdouts = _source_families(target_references, holdout_references)
    category_of = {
        str(ref["name"]): family
        for families in (targets, holdouts)
        for family, (_label, members) in families.items()
        for ref in members
        if isinstance(ref.get("name"), str)
    }
    active_aliases = sorted(_calibration_rows(diagnostics))
    active_families = sorted(
        {
            str(category_of[alias])
            for alias in active_aliases
            if isinstance(category_of.get(alias), str)
        }
    )
    name = source_coverage_name(country)
    result = source_coverage_gate(
        _coverage_entries(targets, holdouts),
        active_target_aliases=active_aliases,
        active_target_families=active_families,
        name=name,
    )
    details = _json_ready(result.details)
    return {
        "schema_version": SOURCE_COVERAGE_SCHEMA_VERSION,
        "classification": "release_gate",
        "country": country,
        "source_contract": {"name": name, "country": country, **source_contract},
        "gate": {
            "name": result.name,
            "passed": result.passed,
            "failures": list(result.failures),
        },
        "coverage_summary": details["coverage_summary"],
        "hard_target_families": details["hard_target_families"],
        "validation_only_families": details["validation_only_families"],
        "source_gap_families": details["source_gap_families"],
        "active_target_aliases": details["active_target_aliases"],
        "active_target_families": details["active_target_families"],
        "missing_hard_targets": details["missing_hard_targets"],
        "reviewed_exclusions": details["reviewed_exclusions"],
        "validation_only_activated": details["validation_only_activated"],
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _sum(values: Iterable[float | None]) -> float | None:
    """The sum, or None when any value is unknown: never a partial sum."""
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _benchmark_source(references: Iterable[Mapping]) -> str | None:
    sources = []
    for reference in references:
        metadata = reference.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        publisher = metadata.get("publisher")
        snapshot = metadata.get("source_snapshot")
        parts = [
            part for part in (publisher, snapshot) if isinstance(part, str) and part
        ]
        if parts:
            source = ", ".join(parts)
            if source not in sources:
                sources.append(source)
    return "; ".join(sources) if sources else None


def demographics_payload(
    *,
    country: str,
    target_references: Mapping,
    diagnostics: Mapping,
    period: int,
) -> dict[str, object]:
    """``demographics.json``: the build's demography cells and age bands.

    A cell is one executable reference of the :data:`DEMOGRAPHY_FAMILY`
    family: its geography and dimensions from the spec, and its calibrated
    population and benchmark from the build's calibration row. A cell the
    build has no row for is unmeasured (None), and so are the totals then.
    The age bands sum the cells by their :data:`AGE_BAND_DIMENSION` value,
    in the order the spec first names each.
    """
    references = [
        ref
        for ref in _references(target_references)
        if ref.get("family") == DEMOGRAPHY_FAMILY and _executable(ref)
    ]
    rows = _calibration_rows(diagnostics)
    cells: list[dict[str, object]] = []
    for reference in references:
        selector = reference.get("ledger_selector", {})
        selector = selector if isinstance(selector, Mapping) else {}
        dimensions = selector.get("dimension_values", {})
        row = rows.get(str(reference["name"]), {})
        population = _number(row.get("final_estimate"))
        benchmark = _number(row.get("target"))
        relative_error = _ratio(
            None if population is None or benchmark is None else population - benchmark,
            benchmark,
        )
        cells.append(
            {
                "target": reference["name"],
                "geography_level": selector.get("geography_level"),
                "geography_id": selector.get("geography_id"),
                "dimensions": dict(dimensions)
                if isinstance(dimensions, Mapping)
                else {},
                "population": population,
                "benchmark": benchmark,
                "relative_error": relative_error,
            }
        )
    total = _sum(cell["population"] for cell in cells) if cells else None
    benchmark_total = _sum(cell["benchmark"] for cell in cells) if cells else None
    bands: dict[str, list[dict[str, object]]] = {}
    for cell in cells:
        band = cell["dimensions"].get(AGE_BAND_DIMENSION)
        if isinstance(band, str):
            bands.setdefault(band, []).append(cell)
    age_bands = []
    for label, members in bands.items():
        population = _sum(cell["population"] for cell in members)
        benchmark = _sum(cell["benchmark"] for cell in members)
        age_bands.append(
            {
                "label": label,
                "population": population,
                "share": _ratio(population, total),
                "benchmark": benchmark,
                "benchmark_share": _ratio(benchmark, benchmark_total),
                "relative_error": _ratio(
                    None
                    if population is None or benchmark is None
                    else population - benchmark,
                    benchmark,
                ),
            }
        )
    return _json_ready(
        {
            "schema_version": DEMOGRAPHICS_SCHEMA_VERSION,
            "country": country,
            "period": int(period),
            "measure": "person_weight",
            # The cells are calibration targets, so their fit is in-sample.
            "in_sample": True,
            "total_population": total,
            "benchmark_total_population": benchmark_total,
            "benchmark_source": _benchmark_source(references),
            "age_bands": age_bands,
            "cells": cells,
        }
    )


def reform_validation_payload(*, country: str, period: int) -> dict[str, object]:
    """``reform_validation.json`` with no reforms: none is scored yet.

    ``out_of_sample_simulated`` is true, as the US writer records it when
    there is no out-of-sample reform: no reform was left unsimulated. The
    publisher refuses a false flag, which means out-of-sample reforms were
    skipped (``microcosm.data.release.publication_refusal``).
    """
    return {
        "schema_version": REFORM_VALIDATION_SCHEMA_VERSION,
        "country": country,
        "baseline_period": int(period),
        "scoring_window": None,
        "out_of_sample_simulated": True,
        "reforms": [],
    }
