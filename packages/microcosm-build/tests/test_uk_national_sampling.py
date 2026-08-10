"""UK scale-ladder sampling policy (#627): clone families, strata, quota."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.national_sampling import (
    sample_uk_national_frame,
    uk_canonical_family_units,
)
from microcosm.frame import Frame


def _families_frame(
    families: list[dict[str, object]],
    *,
    clone_levels: int = 1,
    clone_region: str = "scotland",
) -> Frame:
    """Build a clone-structured UK national frame from family descriptors.

    Each family descriptor carries ``id`` (canonical household id),
    ``region``, ``spi``, ``cg``, and ``weight``.  Every family gets one
    person per household row (person id = household id + 200, so the
    canonical max — and therefore the clone multiplier — is person-driven,
    mirroring the real arithmetic) and ``clone_levels`` geography clones
    whose region is uniform per clone level.
    """

    canonical_max = max(int(family["id"]) for family in families) + 200
    multiplier = 10 ** max(1, len(str(canonical_max)))
    household_rows: list[dict[str, object]] = []
    for family in families:
        for level in range(clone_levels + 1):
            household_rows.append(
                {
                    "household_id": int(family["id"]) + level * multiplier,
                    "clone_index": level,
                    "region": family["region"] if level == 0 else clone_region,
                    "household_is_spi_synthetic": bool(family["spi"]),
                    "household_is_capital_gains_clone": bool(family["cg"]),
                    "household_weight": float(family["weight"]),
                }
            )
    household_rows.sort(key=lambda row: row["household_id"])
    household = pd.DataFrame(household_rows)
    household_ids = household["household_id"].to_numpy(dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": household_ids + 200,
            "person_benunit_id": household_ids + 5_000_000,
            "person_household_id": household_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": household_ids + 5_000_000})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def _standard_families() -> list[dict[str, object]]:
    return [
        {"id": 101, "region": "london", "spi": False, "cg": False, "weight": 10.0},
        {"id": 102, "region": "north", "spi": False, "cg": False, "weight": 10.0},
        {"id": 103, "region": "london", "spi": False, "cg": False, "weight": 10.0},
        {"id": 104, "region": "north", "spi": False, "cg": False, "weight": 10.0},
        {"id": 105, "region": "london", "spi": False, "cg": False, "weight": 10.0},
        {"id": 106, "region": "north", "spi": False, "cg": False, "weight": 10.0},
        {"id": 107, "region": "london", "spi": True, "cg": False, "weight": 0.0},
        {"id": 108, "region": "north", "spi": True, "cg": False, "weight": 0.0},
    ]


def _family_of(household_id: int, multiplier: int) -> int:
    return household_id % multiplier if household_id >= multiplier else household_id


def test_family_units_mirror_the_stage_fence_arithmetic() -> None:
    frame = _families_frame(_standard_families())
    units, forced, multiplier = uk_canonical_family_units(frame)
    # canonical_max = 108 + 200 = 308 -> three digits -> multiplier 1000.
    assert multiplier == 1_000
    household_ids = frame.table("household")["household_id"].to_numpy()
    clone_index = frame.table("household")["clone_index"].to_numpy()
    np.testing.assert_array_equal(units, household_ids - clone_index * multiplier)
    # The argmax canonical household (108) and the argmax canonical person's
    # household (also 108) pin the multiplier's digit count.
    assert forced == (108,)


def test_non_reversing_clone_fails_closed() -> None:
    families = _standard_families()
    frame = _families_frame(families)
    household = frame.table("household").copy()
    # A clone whose reversal lands on no canonical row: corrupt one clone id.
    clone_position = int(np.argmax(household["clone_index"].to_numpy() == 1))
    household.loc[household.index[clone_position], "household_id"] = 1_999
    person = frame.table("person").copy()
    person.loc[
        person["person_household_id"]
        == frame.table("household")["household_id"].iloc[clone_position],
        "person_household_id",
    ] = 1_999
    person = person.sort_values("person_household_id").reset_index(drop=True)
    household = household.sort_values("household_id").reset_index(drop=True)
    benunit = frame.table("benunit")
    broken = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )
    with pytest.raises(ValueError, match="do not reverse"):
        uk_canonical_family_units(broken)


def test_sampling_keeps_families_whole_and_retains_the_argmax_family() -> None:
    frame = _families_frame(_standard_families())
    for seed in (0, 1, 7, 42):
        sampled, receipt = sample_uk_national_frame(frame, fraction=0.5, seed=seed)
        sampled_ids = sampled.table("household")["household_id"].tolist()
        sampled_units = {_family_of(value, 1_000) for value in sampled_ids}
        for unit in sampled_units:
            assert unit in sampled_ids
            assert unit + 1_000 in sampled_ids
        # Forced argmax retention: family 108 is in a floor-zero stratum
        # (one spi/north family at 0.5), so only the forced union keeps it.
        assert 108 in sampled_units
        assert receipt["forced_unit_inclusions"]["forced_unit_count"] == 1
        assert receipt["uk_policy"]["clone_multiplier"] == 1_000


def test_sampled_frame_is_normalized_refreshed_and_valid() -> None:
    frame = _families_frame(_standard_families())
    full_mass = float(frame.weights_for("household").total)
    sampled, receipt = sample_uk_national_frame(frame, fraction=0.5, seed=3)
    validate_uk_national_frame(sampled)
    weights = sampled.weights_for("household")
    assert float(weights.total) == pytest.approx(full_mass)
    np.testing.assert_array_equal(
        sampled.table("household")["household_weight"].to_numpy(dtype="float64"),
        weights.values,
    )
    record = sampled.mass_log[-1]
    assert record.entity == "household"
    assert "composition-preserving" in record.reason
    assert receipt["normalization_factor"] == pytest.approx(
        full_mass / float(receipt["sampled_household_mass"])
    )
    assert receipt["normalized_household_mass"] == pytest.approx(full_mass)


def test_strata_are_proportional_per_channel_and_canonical_region() -> None:
    frame = _families_frame(_standard_families())
    _, receipt = sample_uk_national_frame(frame, fraction=0.5, seed=11)
    strata = receipt["strata"]
    assert strata["spi=False|cg=False|region=london"]["eligible_units"] == 3
    assert strata["spi=False|cg=False|region=london"]["requested_units"] == 1
    assert strata["spi=False|cg=False|region=north"]["eligible_units"] == 3
    assert strata["spi=True|cg=False|region=london"]["requested_units"] == 0
    # The forced argmax family lands in the spi/north stratum beyond its
    # floor-zero request.
    assert strata["spi=True|cg=False|region=north"]["realized_units"] == 1


def test_spi_quota_violation_fails_closed() -> None:
    families = [
        {"id": 101, "region": "london", "spi": False, "cg": False, "weight": 10.0},
        {"id": 102, "region": "london", "spi": True, "cg": False, "weight": 0.0},
        {"id": 103, "region": "london", "spi": True, "cg": False, "weight": 0.0},
        {"id": 104, "region": "north", "spi": False, "cg": False, "weight": 10.0},
    ]
    frame = _families_frame(families)
    # At 0.5 the base/london stratum floors to zero while an spi/london
    # family is drawn (or force-retained), so the sampled artifact carries a
    # clone-0 london cell with dead > base — the post-sample quota check
    # must refuse it.
    with pytest.raises(ValueError, match="SPI replacement quota"):
        sample_uk_national_frame(frame, fraction=0.5, seed=5)


def test_missing_lineage_columns_fail_closed() -> None:
    frame = _families_frame(_standard_families())
    household = frame.table("household").drop(columns=["clone_index"])
    stripped = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2023",
    )
    with pytest.raises(ValueError, match="missing \\['clone_index'\\]"):
        sample_uk_national_frame(stripped, fraction=0.5, seed=1)


def test_full_fraction_is_a_structural_no_op() -> None:
    frame = _families_frame(_standard_families())
    sampled, receipt = sample_uk_national_frame(frame, fraction=1.0, seed=42)
    assert sampled is frame
    assert "normalization_factor" not in receipt
    assert receipt["uk_policy"]["spi_replacement_quota_checked"] is True
