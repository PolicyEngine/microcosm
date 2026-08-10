"""The shared scale-ladder sampler (#627): promoted US draw + unit/strata policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_sampling import (
    EXACT_COUNT_RULE,
    ids_sha256,
    normalize_sampled_household_mass,
    sample_frame_households,
    validate_sample_fraction,
    validate_sample_seed,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

UK_SHAPED_SCHEMA = EntitySchema(group_entities=("benunit", "household"))

#: Digests captured from the pre-promotion ``us_runtime.stacked_spine``
#: sampler over household ids 101..140 — the RNG-stream preservation pin.
#: The draw depends only on the sorted household-id inventory and the seed,
#: never on the schema, so a UK-shaped frame must reproduce them exactly.
_PINNED_DRAWS = {
    (0.55, 7): {
        "requested": 22,
        "sha256": "56fa918399d550c375c3f1b11c96825a01dfc6da0b6bb8be5b78c149ef42d6cc",
    },
    (0.10, 578): {
        "requested": 4,
        "sha256": "7b392a2144ccfb2e6f23faa41c483365d7573029a4de26ad3cec03fd0cb9354c",
        "selected_household_ids": [109, 114, 131, 134],
    },
}


def _uk_shaped_frame(
    household_ids: list[int],
    *,
    weights: list[float] | None = None,
    persons_per_household: dict[int, int] | None = None,
    household_columns: dict[str, list[object]] | None = None,
) -> Frame:
    persons_per_household = persons_per_household or {}
    household_array = np.asarray(household_ids, dtype=np.int64)
    person_household: list[int] = []
    for household_id in household_ids:
        person_household.extend(
            [household_id] * persons_per_household.get(household_id, 1)
        )
    person_household_array = np.asarray(person_household, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(person_household_array) + 1, dtype=np.int64),
            "person_benunit_id": person_household_array + 500_000,
            "person_household_id": person_household_array,
        }
    )
    household = pd.DataFrame({"household_id": household_array})
    for column, values in (household_columns or {}).items():
        household[column] = values
    benunit = pd.DataFrame({"benunit_id": household_array + 500_000})
    weight_values = (
        np.linspace(50.0, 150.0, len(household_ids))
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    return Frame(
        tables={"person": person, "benunit": benunit, "household": household},
        schema=UK_SHAPED_SCHEMA,
        weights={"household": Weights(weight_values, WeightKind.DESIGN)},
    )


def _clone_family_frame() -> tuple[Frame, np.ndarray, np.ndarray]:
    """Ten canonical households plus one clone each; returns units and strata."""

    canonical = list(range(101, 111))
    clones = [value + 1_000 for value in canonical]
    household_ids = canonical + clones
    frame = _uk_shaped_frame(sorted(household_ids))
    ordered = np.asarray(sorted(household_ids), dtype=np.int64)
    units = np.where(ordered >= 1_000, ordered - 1_000, ordered)
    strata = np.asarray(
        ["spi" if unit >= 107 else "base" for unit in units], dtype=object
    )
    return frame, units, strata


def test_default_draw_matches_promoted_us_digests() -> None:
    frame = _uk_shaped_frame(list(range(101, 141)))
    for (fraction, seed), pinned in _PINNED_DRAWS.items():
        sampled, receipt = sample_frame_households(
            frame, fraction=fraction, seed=seed, source_name="UK"
        )
        assert receipt["eligible_household_count"] == 40
        assert receipt["requested_household_count"] == pinned["requested"]
        assert receipt["realized_household_count"] == pinned["requested"]
        assert receipt["selected_household_ids_sha256"] == pinned["sha256"]
        if "selected_household_ids" in pinned:
            assert (
                sampled.table("household")["household_id"].tolist()
                == pinned["selected_household_ids"]
            )


def test_default_receipt_shape_matches_promoted_fields() -> None:
    frame = _uk_shaped_frame(list(range(101, 141)))
    _, receipt = sample_frame_households(frame, fraction=0.5, seed=1, source_name="UK")
    assert list(receipt) == [
        "fraction",
        "seed",
        "eligible_household_count",
        "requested_household_count",
        "realized_household_count",
        "exact_count_rule",
        "selected_household_ids_sha256",
        "incoming_household_mass",
        "sampled_household_mass",
    ]
    assert receipt["exact_count_rule"] == EXACT_COUNT_RULE


def test_full_fraction_is_a_no_op() -> None:
    frame = _uk_shaped_frame(list(range(101, 111)))
    sampled, receipt = sample_frame_households(
        frame, fraction=1.0, seed=42, source_name="UK"
    )
    assert sampled is frame
    assert receipt["realized_household_count"] == 10
    assert receipt["requested_household_count"] == 10
    assert "sampling_unit" not in receipt


def test_floors_to_zero_fails_closed() -> None:
    frame = _uk_shaped_frame(list(range(101, 111)))
    with pytest.raises(ValueError, match="floors to zero"):
        sample_frame_households(frame, fraction=0.05, seed=1, source_name="UK")


def test_whole_unit_selection_takes_clone_families_together() -> None:
    frame, units, _ = _clone_family_frame()
    sampled, receipt = sample_frame_households(
        frame,
        fraction=0.5,
        seed=9,
        source_name="UK",
        unit_ids=units,
        unit_noun="clone family",
    )
    sampled_ids = set(sampled.table("household")["household_id"].tolist())
    sampled_units = {
        value - 1_000 if value >= 1_000 else value for value in sampled_ids
    }
    for unit in sampled_units:
        assert unit in sampled_ids
        assert unit + 1_000 in sampled_ids
    unit_block = receipt["sampling_unit"]
    assert unit_block["noun"] == "clone family"
    assert unit_block["eligible_unit_count"] == 10
    assert unit_block["requested_unit_count"] == 5
    assert unit_block["realized_unit_count"] == 5
    assert "requested_household_count" not in receipt
    assert receipt["realized_household_count"] == 10


def test_strata_draw_is_proportional_per_group() -> None:
    frame, units, strata = _clone_family_frame()
    _, receipt = sample_frame_households(
        frame,
        fraction=0.5,
        seed=9,
        source_name="UK",
        unit_ids=units,
        unit_strata=strata,
    )
    strata_block = receipt["strata"]
    assert strata_block["base"]["eligible_units"] == 6
    assert strata_block["base"]["requested_units"] == 3
    assert strata_block["base"]["realized_units"] == 3
    assert strata_block["spi"]["eligible_units"] == 4
    assert strata_block["spi"]["requested_units"] == 2
    assert strata_block["spi"]["realized_units"] == 2
    assert receipt["sampling_unit"]["realized_unit_count"] == 5


def test_strata_must_be_constant_within_unit() -> None:
    frame, units, strata = _clone_family_frame()
    broken = strata.copy()
    broken[0] = "spi" if broken[0] == "base" else "base"
    with pytest.raises(ValueError, match="constant within each sampling unit"):
        sample_frame_households(
            frame,
            fraction=0.5,
            seed=9,
            source_name="UK",
            unit_ids=units,
            unit_strata=broken,
        )


def test_strata_reject_missing_values() -> None:
    frame, units, strata = _clone_family_frame()
    broken = strata.astype(object)
    broken[3] = None
    with pytest.raises(ValueError, match="missing values"):
        sample_frame_households(
            frame,
            fraction=0.5,
            seed=9,
            source_name="UK",
            unit_ids=units,
            unit_strata=broken,
        )


def test_forced_units_are_added_after_the_draw() -> None:
    frame, units, _ = _clone_family_frame()
    _, unforced = sample_frame_households(
        frame,
        fraction=0.3,
        seed=11,
        source_name="UK",
        unit_ids=units,
    )
    sampled, receipt = sample_frame_households(
        frame,
        fraction=0.3,
        seed=11,
        source_name="UK",
        unit_ids=units,
        forced_unit_ids=(101, 110),
    )
    forced_block = receipt["forced_unit_inclusions"]
    assert forced_block["forced_unit_count"] == 2
    assert forced_block["forced_unit_ids_sha256"] == ids_sha256(np.asarray([101, 110]))
    sampled_ids = set(sampled.table("household")["household_id"].tolist())
    assert {101, 1_101, 110, 1_110} <= sampled_ids
    # Forced units are unioned after the draw, so the drawn selection is a
    # subset of the forced run's selection: the RNG stream is untouched.
    assert (
        receipt["sampling_unit"]["realized_unit_count"]
        == unforced["sampling_unit"]["realized_unit_count"]
        + forced_block["added_beyond_draw_count"]
    )


def test_forced_unit_absent_from_inventory_fails_closed() -> None:
    frame, units, _ = _clone_family_frame()
    with pytest.raises(ValueError, match="absent from the unit inventory"):
        sample_frame_households(
            frame,
            fraction=0.5,
            seed=9,
            source_name="UK",
            unit_ids=units,
            forced_unit_ids=(999,),
        )


def test_normalize_sampled_household_mass_restores_full_mass() -> None:
    frame = _uk_shaped_frame(list(range(101, 141)))
    full_mass = float(frame.weights_for("household").total)
    sampled, _ = sample_frame_households(frame, fraction=0.5, seed=3, source_name="UK")
    normalized, factor = normalize_sampled_household_mass(
        sampled, target_mass=full_mass, source_name="UK"
    )
    assert factor == pytest.approx(
        full_mass / float(sampled.weights_for("household").total)
    )
    assert float(normalized.weights_for("household").total) == pytest.approx(full_mass)
    assert normalized.weights_for("household").kind == WeightKind.DESIGN
    record = normalized.mass_log[-1]
    assert record.entity == "household"
    assert "composition-preserving" in record.reason


def test_sample_fraction_and_seed_validation() -> None:
    for bad_fraction in (0.0, 1.5, float("nan"), True, "0.5"):
        with pytest.raises(ValueError, match="fraction must be a finite number"):
            validate_sample_fraction(bad_fraction)
    with pytest.raises(ValueError, match="boundary: rung fraction"):
        validate_sample_fraction(2.0, label="rung", boundary="boundary")
    for bad_seed in (-1, 1.5, True, "7"):
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            validate_sample_seed(bad_seed)
