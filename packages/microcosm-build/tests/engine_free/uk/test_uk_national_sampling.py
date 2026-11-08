"""UK spine sampling policy (#627): source families, strata, normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.national_sampling import (
    sample_uk_spine_frame,
    uk_spine_source_family_units,
)
from microcosm.frame import Frame


def _spine_family_frame(
    *,
    families_per_region: int = 4,
    orphan_source: int | None = None,
    multi_region_family: bool = False,
    fractional_source: bool = False,
) -> Frame:
    """Build a tiny spine with explicit raw-family lineage and derivatives."""

    raw = [
        *[
            (household_id, "london")
            for household_id in range(1, 1 + families_per_region)
        ],
        *[
            (household_id, "north")
            for household_id in range(
                1 + families_per_region, 1 + 2 * families_per_region
            )
        ],
    ]
    north_source = 1 + families_per_region
    rows = [
        {
            "household_id": household_id,
            "source_household_id": household_id,
            "region": region,
            "support_clone": 0,
            "spi": False,
            "cgt": False,
            "band_donor": False,
            "weight": 10.0,
        }
        for household_id, region in raw
    ]
    rows.extend(
        [
            {
                "household_id": 10_001,
                "source_household_id": 1,
                "region": "london",
                "support_clone": 0,
                "spi": True,
                "cgt": False,
                "band_donor": False,
                "weight": 2.0,
            },
            {
                "household_id": 10_002,
                "source_household_id": 1,
                "region": "london",
                "support_clone": 1,
                "spi": False,
                "cgt": False,
                "band_donor": False,
                "weight": 2.0,
            },
            {
                "household_id": 20_001,
                "source_household_id": north_source,
                "region": "north",
                "support_clone": 0,
                "spi": False,
                "cgt": True,
                "band_donor": False,
                "weight": 2.0,
            },
            {
                "household_id": 20_002,
                "source_household_id": north_source,
                "region": "north",
                "support_clone": 0,
                "spi": False,
                "cgt": False,
                "band_donor": True,
                "weight": 2.0,
            },
        ]
    )
    if orphan_source is not None:
        rows[-1]["source_household_id"] = orphan_source
    if multi_region_family:
        rows.append(
            {
                "household_id": 30_001,
                "source_household_id": 1,
                "region": "north",
                "support_clone": 0,
                "spi": False,
                "cgt": False,
                "band_donor": False,
                "weight": 1.0,
            }
        )
    household = pd.DataFrame(
        {
            "household_id": [row["household_id"] for row in rows],
            "source_household_id": [row["source_household_id"] for row in rows],
            "source_household_key": [
                f"2023:{row['source_household_id']}" for row in rows
            ],
            "household_source_id": [row["source_household_id"] for row in rows],
            "household_support_clone_index": [row["support_clone"] for row in rows],
            "household_is_spi_synthetic": [row["spi"] for row in rows],
            "household_is_capital_gains_clone": [row["cgt"] for row in rows],
            "household_is_cgt_support_copy": [row["band_donor"] for row in rows],
            "household_is_cgt_residential_clone": [
                row.get("residential_arm", False) for row in rows
            ],
            "region": [row["region"] for row in rows],
            "household_weight": [row["weight"] for row in rows],
        }
    )
    if fractional_source:
        household["source_household_id"] = household["source_household_id"].astype(
            float
        )
        household.loc[household.index[-1], "source_household_id"] += 0.5
    household_ids = household["household_id"].to_numpy(dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": household_ids + 1_000,
            "person_benunit_id": household_ids + 2_000,
            "person_household_id": household_ids,
        }
    )
    benunit = pd.DataFrame({"benunit_id": household_ids + 2_000})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2023",
    )


def test_spine_source_units_use_raw_family_regions() -> None:
    frame = _spine_family_frame()
    units, strata = uk_spine_source_family_units(frame)
    household_ids = frame.table("household")["household_id"].to_numpy()
    mapping = dict(zip(household_ids.tolist(), units.tolist(), strict=True))
    assert mapping[10_001] == 1
    assert mapping[10_002] == 1
    assert mapping[20_001] == 5
    assert mapping[20_002] == 5
    assert set(strata[units == 1]) == {"region=london"}
    assert set(strata[units == 5]) == {"region=north"}


@pytest.mark.parametrize("fraction,rung", [(0.01, "f001"), (0.10, "f010")])
def test_spine_sampling_is_stratified_keeps_families_and_normalizes(
    fraction: float,
    rung: str,
) -> None:
    frame = _spine_family_frame(families_per_region=100)
    units, _ = uk_spine_source_family_units(frame)
    household_ids = frame.table("household")["household_id"].to_numpy()
    family_by_household = dict(zip(household_ids.tolist(), units.tolist(), strict=True))
    full_rows_by_family = {
        family: set(household_ids[units == family].tolist())
        for family in np.unique(units)
    }
    full_mass = float(frame.weights_for("household").total)

    sampled, receipt = sample_uk_spine_frame(frame, fraction=fraction, seed=19)

    sampled_ids = set(sampled.table("household")["household_id"].tolist())
    sampled_families = {family_by_household[value] for value in sampled_ids}
    for family in sampled_families:
        assert full_rows_by_family[family] <= sampled_ids
    assert float(sampled.weights_for("household").total) == pytest.approx(full_mass)
    assert receipt["pre_family_count"] == 200
    assert receipt["post_family_count"] == int(200 * fraction)
    assert receipt["strata_count"] == 2
    assert receipt["receipt"]["strata"]["region=london"]["requested_units"] == int(
        100 * fraction
    )
    assert receipt["receipt"]["strata"]["region=north"]["requested_units"] == int(
        100 * fraction
    )
    assert receipt["normalization_factor"] > 0.0
    assert receipt["rung_token"] == rung


def test_spine_sampling_missing_lineage_column_fails_closed() -> None:
    frame = _spine_family_frame()
    household = frame.table("household").drop(columns=["source_household_id"])
    stripped = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2023",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="source_household_id"):
        uk_spine_source_family_units(stripped)


def test_spine_sampling_fractional_source_fails_closed() -> None:
    with pytest.raises(ValueError, match="source_household_id must be integer"):
        uk_spine_source_family_units(_spine_family_frame(fractional_source=True))


def test_spine_sampling_refuses_numeric_region_codes() -> None:
    """Region strata labels are formatted values, so an integer code in one
    build and a float code in another would silently split a stratum; the
    sampler accepts string regions only."""

    frame = _spine_family_frame()
    household = frame.table("household").copy()
    household["region"] = np.arange(len(household)) % 3
    numeric = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2023",
        household_weights=frame.weights_for("household").values,
    )
    with pytest.raises(ValueError, match="region must contain non-empty strings"):
        sample_uk_spine_frame(numeric, fraction=0.5, seed=1)


def test_spine_sampling_orphan_derivative_fails_closed() -> None:
    # The refusals count families and name no source household (microcosm#931).
    with pytest.raises(ValueError) as caught:
        uk_spine_source_family_units(_spine_family_frame(orphan_source=999))
    assert str(caught.value) == (
        "UK spine sample: 1 source_household_id family key(s) lack a "
        "household_id-bearing raw row."
    )


def test_spine_sampling_multi_region_raw_family_fails_closed() -> None:
    with pytest.raises(ValueError) as caught:
        uk_spine_source_family_units(_spine_family_frame(multi_region_family=True))
    assert str(caught.value) == (
        "UK spine sample: 1 source family(ies) span more than one raw-row region."
    )


def test_spine_sampling_counts_the_families_a_sample_left_incomplete(
    monkeypatch,
) -> None:
    from microcosm.build.uk_runtime import national_sampling

    real = national_sampling.uk_spine_source_family_units
    calls: list[int] = []

    def relabel_one_row_after_sampling(frame):
        units, strata = real(frame)
        calls.append(len(units))
        if len(calls) == 2:
            # The first sampled row, a family of one, joins the next sampled
            # family, which then holds a row more than in the full frame.
            units = units.copy()
            units[0] = next(unit for unit in units if unit != units[0])
        return units, strata

    monkeypatch.setattr(
        national_sampling,
        "uk_spine_source_family_units",
        relabel_one_row_after_sampling,
    )
    with pytest.raises(ValueError) as caught:
        sample_uk_spine_frame(
            _spine_family_frame(families_per_region=100), fraction=0.5, seed=19
        )
    assert str(caught.value) == (
        "UK spine sample retained 1 incomplete source family(ies)."
    )
