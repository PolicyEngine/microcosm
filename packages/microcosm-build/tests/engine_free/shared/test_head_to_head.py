"""The country-neutral head-to-head comparison of two scored artifacts."""

from __future__ import annotations

import math

import pytest

from microcosm.build.head_to_head import (
    HEAD_TO_HEAD_SCHEMA_VERSION,
    LOWER_CANDIDATE,
    LOWER_EQUAL,
    LOWER_REFERENCE,
    ScoredTarget,
    compare_scored_targets,
    lower_side,
    scored_targets_from_mapping,
)


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    [
        (0.1, 0.2, LOWER_CANDIDATE),
        (-0.1, 0.2, LOWER_CANDIDATE),
        (0.3, -0.2, LOWER_REFERENCE),
        (0.2, -0.2, LOWER_EQUAL),
        (0.0, -0.0, LOWER_EQUAL),
        (math.nan, 0.1, LOWER_EQUAL),
        (0.1, math.nan, LOWER_EQUAL),
    ],
)
def test_the_lower_side_is_decided_on_magnitude(candidate, reference, expected) -> None:
    assert lower_side(candidate, reference) == expected


def _scored(errors, families=None):
    return [
        ScoredTarget(
            name=name,
            relative_error=error,
            period=2025,
            family=None if families is None else families[name],
        )
        for name, error in errors.items()
    ]


def test_the_comparison_counts_targets_and_the_loss_gap() -> None:
    families = {"a": "income", "b": "income", "c": "benefits", "d": "benefits"}
    head_to_head = compare_scored_targets(
        _scored({"a": 0.01, "b": -0.5, "c": 0.2, "d": 0.3}, families),
        _scored({"a": -0.02, "b": 0.1, "c": -0.2, "d": 0.1}),
        candidate_loss=0.25,
        reference_loss=0.75,
    )

    assert head_to_head.counts == {
        "candidate_lower_count": 1,
        "reference_lower_count": 2,
        "equal_count": 1,
    }
    assert head_to_head.counts_by_family() == {
        "benefits": {
            "candidate_lower_count": 0,
            "reference_lower_count": 1,
            "equal_count": 1,
        },
        "income": {
            "candidate_lower_count": 1,
            "reference_lower_count": 1,
            "equal_count": 0,
        },
    }
    payload = head_to_head.to_payload()
    assert payload["schema_version"] == HEAD_TO_HEAD_SCHEMA_VERSION
    assert payload["decision"] == "owner_decides"
    assert payload["no_threshold_applied"] is True
    assert payload["loss"] == {
        "candidate": 0.25,
        "reference": 0.75,
        "candidate_minus_reference": -0.5,
    }
    first = payload["targets"]["compared"][0]
    assert first == {
        "name": "a",
        "period": 2025,
        "family": "income",
        "candidate_relative_error": 0.01,
        "reference_relative_error": -0.02,
        "candidate_absolute_relative_error": 0.01,
        "reference_absolute_relative_error": 0.02,
        "candidate_minus_reference_absolute_relative_error": 0.01 - 0.02,
        "lower_absolute_relative_error": LOWER_CANDIDATE,
    }
    assert "verdict" not in payload


def test_targets_without_a_family_are_counted_last() -> None:
    head_to_head = compare_scored_targets(
        [ScoredTarget("a", 0.1, family="x"), ScoredTarget("b", 0.1)],
        [ScoredTarget("a", 0.2), ScoredTarget("b", 0.2)],
        candidate_loss=1.0,
        reference_loss=1.0,
    )

    assert list(head_to_head.counts_by_family()) == ["x", None]


def test_different_surfaces_are_refused() -> None:
    with pytest.raises(ValueError, match="different numbers of targets"):
        compare_scored_targets(
            [ScoredTarget("a", 0.1)], [], candidate_loss=0.0, reference_loss=0.0
        )
    with pytest.raises(ValueError, match="contracts differ"):
        compare_scored_targets(
            [ScoredTarget("a", 0.1, period=2025)],
            [ScoredTarget("a", 0.1, period=2024)],
            candidate_loss=0.0,
            reference_loss=0.0,
        )
    with pytest.raises(ValueError, match="belongs to family"):
        compare_scored_targets(
            [ScoredTarget("a", 0.1, family="x")],
            [ScoredTarget("a", 0.1, family="y")],
            candidate_loss=0.0,
            reference_loss=0.0,
        )


def test_scored_targets_from_a_mapping_follow_the_given_order() -> None:
    scored = scored_targets_from_mapping(
        {"b": 0.2, "a": 0.1}, families={"a": "x"}, order=["b", "a"]
    )

    assert [(row.name, row.family) for row in scored] == [("b", None), ("a", "x")]
    assert [row.name for row in scored_targets_from_mapping({"b": 0.2, "a": 0.1})] == [
        "a",
        "b",
    ]
    with pytest.raises(ValueError, match="No score for targets"):
        scored_targets_from_mapping({"a": 0.1}, order=["a", "z"])
