"""UK release-gate country helpers."""

from __future__ import annotations

import pytest

from populace.build.uk import (
    UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
    UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS,
    UK_REVIEWED_EXPORT_EXCLUSIONS,
    uk_export_surface_gate,
    uk_release_gate_report,
    uk_target_fit_gate,
    uk_target_surface_gate,
)


def test_uk_export_surface_allows_reviewed_candidate_extras() -> None:
    reference = {
        "person.age",
        "person.employment_income",
        "household.household_weight",
    }
    candidate = reference | set(UK_ALLOWED_EXTRA_EXPORT_COLUMNS)

    result = uk_export_surface_gate(candidate, reference)

    assert result.passed
    assert result.details["unused_allowed_extra_columns"] == []


def test_uk_export_surface_accepts_candidate_iterators() -> None:
    result = uk_export_surface_gate(iter(["person.age"]), {"person.age"})

    assert result.passed
    assert result.details["candidate_columns"] == 1


def test_uk_export_surface_keeps_missing_incumbent_columns_hard() -> None:
    reference = {
        "person.age",
        *UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS,
        *UK_REVIEWED_EXPORT_EXCLUSIONS,
    }
    candidate = {"person.age"}

    result = uk_export_surface_gate(candidate, reference)

    assert not result.passed
    for column in UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS:
        assert column in result.details["missing_reference_columns"]
    for column in UK_REVIEWED_EXPORT_EXCLUSIONS:
        assert column not in result.details["missing_reference_columns"]


def test_uk_export_surface_records_dead_zero_reference_exclusion() -> None:
    reference = {"person.age", *UK_REVIEWED_EXPORT_EXCLUSIONS}
    candidate = {"person.age"}

    result = uk_export_surface_gate(candidate, reference)

    assert result.passed
    assert result.details["reviewed_exclusions"] == dict(UK_REVIEWED_EXPORT_EXCLUSIONS)


def test_uk_export_surface_rejects_exported_dead_zero_reference_exclusion() -> None:
    reference = {"person.age", *UK_REVIEWED_EXPORT_EXCLUSIONS}
    candidate = set(reference)

    result = uk_export_surface_gate(candidate, reference)

    assert not result.passed
    assert result.details["forbidden_candidate_columns"] == sorted(
        UK_REVIEWED_EXPORT_EXCLUSIONS
    )


def test_uk_export_surface_accepts_explicit_reviewed_exclusion() -> None:
    reference = {"person.age", "person.legacy_reference_only"}
    candidate = {"person.age"}

    result = uk_export_surface_gate(
        candidate,
        reference,
        reviewed_exclusions={
            "person.legacy_reference_only": "Formula-owned in the replacement build."
        },
    )

    assert result.passed
    assert result.details["reviewed_exclusions"] == {
        "person.legacy_reference_only": "Formula-owned in the replacement build."
    }


def test_uk_export_surface_refuses_to_waive_hard_required_columns() -> None:
    with pytest.raises(ValueError, match="hard-required reference columns"):
        uk_export_surface_gate(
            {"person.age"},
            {"person.age", *UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS},
            reviewed_exclusions={
                UK_KNOWN_MISSING_REFERENCE_EXPORT_COLUMNS[0]: "Do not allow this."
            },
        )


def test_uk_target_surface_must_cover_reference_targets() -> None:
    result = uk_target_surface_gate(
        ["ons/population"],
        ["ons/population", "hmrc/income_tax"],
    )

    assert not result.passed
    assert result.details["missing_reference_targets"] == ["hmrc/income_tax"]


def test_uk_target_surface_has_no_reviewed_exclusion_escape_hatch() -> None:
    with pytest.raises(TypeError):
        uk_target_surface_gate(  # type: ignore[call-arg]
            ["ons/population"],
            ["ons/population", "hmrc/income_tax"],
            reviewed_exclusions={"hmrc/income_tax": "No waiver."},
        )


def test_uk_release_report_combines_export_and_target_surface() -> None:
    report = uk_release_gate_report(
        candidate_columns={"person.age"},
        reference_columns={"person.age"},
        candidate_targets={"ons/population"},
        reference_targets={"ons/population"},
    )

    assert report.passed
    assert sorted(report.to_manifest()["gates"]) == [
        "export_surface",
        "target_surface",
    ]


def test_uk_target_fit_gate_fails_large_shipped_target_errors() -> None:
    result = uk_target_fit_gate(
        {
            "ons/couple_no_children_households": 0.73,
            "ons/lone_households_under_65": 0.02,
        }
    )

    assert not result.passed
    assert result.details["failing_targets"] == {
        "ons/couple_no_children_households": 0.73
    }


def test_uk_target_fit_gate_allows_reviewed_exclusions() -> None:
    result = uk_target_fit_gate(
        {"ons/known_proxy_gap": -0.50},
        reviewed_exclusions={"ons/known_proxy_gap": "Documented proxy limitation."},
    )

    assert result.passed
    assert result.details["reviewed_exclusions"] == {
        "ons/known_proxy_gap": "Documented proxy limitation."
    }
