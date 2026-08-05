"""Stacked-spine pilot: stack, doctrine, gap-fill, gates (populace#578 revision).

The ratified increment-2 revision replaces the two-spine agreement construct
with ONE origin-labeled spine: ASEC plus a seeded ACS household sample,
cross-origin gap-fill with native predictors, a single PUF pass after
gap-fill, a pre-simulation completeness gate, and a by-origin battery with
per-family declared metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.stacked_spine import (
    DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES,
    STACKED_SPINE_MANIFEST_KEY,
    assemble_stacked_spine,
    sample_acs_households,
    validate_stacked_spine_frame,
)
from populace.build.us_runtime.support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _source_frame(
    *,
    household_ids: list[int],
    persons_per_household: dict[int, int] | None = None,
    weights: list[float],
    extra_person_columns: dict[str, float | str] | None = None,
    extra_household_columns: dict[str, object] | None = None,
    stratum: str,
) -> Frame:
    """Build one pre-assembly source frame of one-tax-unit households."""

    persons_per_household = persons_per_household or {}
    person_household: list[int] = []
    for household_id in household_ids:
        person_household.extend(
            [household_id] * persons_per_household.get(household_id, 1)
        )
    person_count = len(person_household)
    person_household_array = np.asarray(person_household, dtype=np.int64)
    group_offsets = {
        "tax_unit": 100_000,
        "spm_unit": 200_000,
        "family": 300_000,
        "marital_unit": 400_000,
    }
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, person_count + 1, dtype=np.int64)
            + household_ids[0] * 1_000,
            "person_household_id": person_household_array,
            "age": np.linspace(25.0, 70.0, person_count),
        }
    )
    for group, offset in group_offsets.items():
        person[f"person_{group}_id"] = person_household_array + offset
    for column, value in (extra_person_columns or {}).items():
        person[column] = value

    household_array = np.asarray(household_ids, dtype=np.int64)
    household = pd.DataFrame(
        {
            "household_id": household_array,
            "state_fips": np.full(len(household_ids), 6, dtype=np.int64),
        }
    )
    for column, value in (extra_household_columns or {}).items():
        household[column] = value
    tables: dict[str, pd.DataFrame] = {"person": person, "household": household}
    for group, offset in group_offsets.items():
        tables[group] = pd.DataFrame({f"{group}_id": household_array + offset})
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
        pd.Series([stratum] * person_count, dtype=object),
    )


def _asec_source() -> Frame:
    return _source_frame(
        household_ids=[11, 12],
        persons_per_household={11: 2},
        weights=[300.0, 100.0],
        extra_person_columns={"asec_detail_income": 40.0},
        stratum="asec_2024",
    )


def _acs_source() -> Frame:
    return _source_frame(
        household_ids=list(range(101, 111)),
        persons_per_household={103: 3, 107: 2},
        weights=[float(10 * position) for position in range(1, 11)],
        extra_person_columns={"acs_native_aggregate": 15.0},
        extra_household_columns={"puma": "0600101"},
        stratum="acs_2024_1yr",
    )


def test_stacked_assembly_is_deterministic_and_floor_exact() -> None:
    first = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    repeated = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )

    sample = first.receipt["acs_sample"]
    assert sample["eligible_household_count"] == 10
    assert sample["requested_household_count"] == 2
    assert sample["realized_household_count"] == 2
    assert sample["exact_count_rule"] == "floor(fraction * eligible)"
    assert first.receipt["acs_sample_fraction"] == 0.25
    assert first.receipt["acs_sample_seed"] == 578
    assert (
        first.receipt["acs_sample"]["selected_household_ids_sha256"]
        == repeated.receipt["acs_sample"]["selected_household_ids_sha256"]
    )
    for entity in first.frame.entities:
        assert_frame_equal(first.frame.table(entity), repeated.frame.table(entity))
    np.testing.assert_array_equal(
        first.frame.weights_for("household").values,
        repeated.frame.weights_for("household").values,
    )

    manifest = first.frame.metadata[STACKED_SPINE_MANIFEST_KEY]
    assert manifest["acs_sample_fraction"] == 0.25
    assert manifest["acs_sample_seed"] == 578
    assert validate_stacked_spine_frame(
        first.frame,
        boundary="determinism fixture",
    )


def test_stacked_assembly_seed_and_fraction_bind_identity() -> None:
    base = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    changed_seed = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=579,
    )
    wider_fraction = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.55,
        acs_sample_seed=578,
    )

    assert (
        base.receipt["acs_sample"]["selected_household_ids_sha256"]
        != changed_seed.receipt["acs_sample"]["selected_household_ids_sha256"]
    )
    assert wider_fraction.receipt["acs_sample"]["realized_household_count"] == 5


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda manifest: manifest["acs_sample"].__setitem__(
                "realized_household_count", 3
            ),
            "realized household count",
        ),
        (
            lambda manifest: manifest["acs_sample"].__setitem__(
                "selected_household_ids_sha256", "0" * 64
            ),
            "selection digest",
        ),
        (
            lambda manifest: manifest.__setitem__("acs_sample_fraction", 0.35),
            "violates floor",
        ),
        (
            lambda manifest: manifest.__setitem__("acs_sample_seed", "578"),
            "acs_sample_seed",
        ),
        (
            lambda manifest: manifest.pop("acs_sample"),
            "sample receipt",
        ),
    ),
)
def test_stacked_manifest_mutations_fail_closed(mutate, match) -> None:
    result = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    stacked = result.frame
    manifest = {
        key: (
            {
                nested_key: (
                    dict(nested_value)
                    if isinstance(nested_value, dict)
                    else nested_value
                )
                for nested_key, nested_value in value.items()
            }
            if isinstance(value, dict)
            else value
        )
        for key, value in result.receipt.items()
    }
    mutate(manifest)
    tampered = Frame(
        {entity: stacked.table(entity) for entity in stacked.entities},
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata={**stacked.metadata, STACKED_SPINE_MANIFEST_KEY: manifest},
    )

    with pytest.raises(ValueError, match=match):
        validate_stacked_spine_frame(tampered, boundary="tampered fixture")


def test_sample_acs_households_takes_whole_lineages() -> None:
    acs = _acs_source()
    sampled, receipt = sample_acs_households(acs, fraction=0.55, seed=7)

    assert receipt["requested_household_count"] == 5
    selected_households = set(sampled.table("household")["household_id"].tolist())
    person = sampled.table("person")
    assert set(person["person_household_id"]) == selected_households
    full_person = acs.table("person")
    for household_id in selected_households:
        expected = int((full_person["person_household_id"] == household_id).sum())
        actual = int((person["person_household_id"] == household_id).sum())
        assert actual == expected
    for group in ("tax_unit", "spm_unit", "family", "marital_unit"):
        assert set(sampled.table(group)[f"{group}_id"]) == set(
            person[f"person_{group}_id"]
        )


def test_sample_floor_zero_fails_closed() -> None:
    with pytest.raises(ValueError, match="floors to zero"):
        sample_acs_households(_acs_source(), fraction=0.05, seed=1)


def test_sample_rejects_provenance_carrying_source() -> None:
    stacked = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
    ).frame
    with pytest.raises(ValueError, match="already carries support provenance"):
        sample_acs_households(stacked, fraction=0.5, seed=1)


def test_weight_harmonization_matches_share_math() -> None:
    asec = _asec_source()
    acs = _acs_source()
    result = assemble_stacked_spine(
        asec,
        acs,
        acs_sample_fraction=0.25,
        acs_sample_seed=578,
    )
    stacked = result.frame
    anchor_mass = float(asec.weights_for("household").total)

    household = stacked.table("household")
    weights = stacked.weights_for("household").values
    channel = household[support_channel_column("household")]
    asec_mass = float(weights[channel.eq("asec").to_numpy()].sum())
    acs_mass = float(weights[channel.eq("acs").to_numpy()].sum())
    assert np.isclose(asec_mass, 0.5 * anchor_mass, rtol=1e-12)
    assert np.isclose(acs_mass, 0.5 * anchor_mass, rtol=1e-12)
    assert float(stacked.weights_for("household").total) == anchor_mass

    harmonization = result.receipt["weight_harmonization"]
    sampled_mass = result.receipt["acs_sample"]["sampled_household_mass"]
    assert np.isclose(
        harmonization["acs"]["scale_factor"],
        0.5 * anchor_mass / sampled_mass,
        rtol=1e-12,
    )
    assert harmonization["asec"]["incoming_mass"] == anchor_mass
    assert np.isclose(
        harmonization["asec"]["scale_factor"],
        0.5,
        rtol=1e-12,
    )

    acs_weights = weights[channel.eq("acs").to_numpy()]
    source_ids = household.loc[
        channel.eq("acs").to_numpy(),
        spine_source_id_column("household"),
    ].to_numpy()
    incoming_by_id = dict(
        zip(
            acs.table("household")["household_id"].tolist(),
            acs.weights_for("household").values.tolist(),
            strict=True,
        )
    )
    incoming = np.asarray(
        [incoming_by_id[int(value)] for value in source_ids],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        acs_weights,
        incoming * harmonization["acs"]["scale_factor"],
        rtol=1e-9,
    )


def test_fraction_one_matches_plain_assembly() -> None:
    asec = _asec_source()
    acs = _acs_source()
    result = assemble_stacked_spine(
        asec,
        acs,
        acs_sample_fraction=1.0,
        acs_sample_seed=123,
    )
    plain = assemble_spines(
        {"asec": _asec_source(), "acs": _acs_source()},
        household_mass_shares=dict(DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES),
        mass_anchor_channel="asec",
    )

    sample = result.receipt["acs_sample"]
    assert sample["realized_household_count"] == sample["eligible_household_count"]
    for entity in plain.entities:
        assert_frame_equal(result.frame.table(entity), plain.table(entity))
    np.testing.assert_array_equal(
        result.frame.weights_for("household").values,
        plain.weights_for("household").values,
    )


def test_selection_digest_uses_raw_spine_ids_under_collision_remap() -> None:
    asec = _source_frame(
        household_ids=[101, 102],
        weights=[300.0, 100.0],
        extra_person_columns={"asec_detail_income": 40.0},
        stratum="asec_2024",
    )
    result = assemble_stacked_spine(
        asec,
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
    )

    household = result.frame.table("household")
    channel = household[support_channel_column("household")]
    clone_index = household[support_clone_index_column("household")]
    native_acs = channel.eq("acs") & clone_index.eq(0)
    remapped = household.loc[native_acs, "household_id"].to_numpy()
    raw = household.loc[native_acs, spine_source_id_column("household")].to_numpy()
    assert not np.array_equal(np.sort(remapped), np.sort(raw))
    assert validate_stacked_spine_frame(
        result.frame,
        boundary="collision fixture",
    )
