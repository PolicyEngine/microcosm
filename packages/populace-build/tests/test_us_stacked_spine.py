"""Stacked-spine pilot: stack, doctrine, gap-fill, gates (populace#578 revision).

The ratified increment-2 revision replaces the two-spine agreement construct
with ONE origin-labeled spine: ASEC plus a seeded ACS household sample,
cross-origin gap-fill with native predictors, a single PUF pass after
gap-fill, a pre-simulation completeness gate, and a by-origin battery with
per-family declared metrics.
"""

from __future__ import annotations

import hashlib
import inspect
import pickle
from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import populace.build.us_runtime.puf_support as puf_support_module
import populace.build.us_runtime.stacked_spine as stacked_spine_module
from populace.build.gates import GateReport, GateResult
from populace.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from populace.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from populace.build.us_runtime.multispine_pool import (
    pool_transfer_target_families,
)
from populace.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
)
from populace.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_CLONE_ATTACHMENT_MANIFEST_KEY,
    PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN,
    clone_us_frame_for_puf_support,
    finalize_us_puf_tax_detail_predictions,
    prepare_us_puf_tax_detail_chain_inputs,
    validate_puf_clone_attachment,
)
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.stacked_spine import (
    DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES,
    STACKED_PILOT_ACS_SAMPLE_FRACTION,
    STACKED_PILOT_ACS_SAMPLE_SEED,
    STACKED_SPINE_MANIFEST_KEY,
    AbsenceProof,
    GapFillDirection,
    OriginBatterySpec,
    assemble_stacked_spine,
    by_origin_battery,
    gap_fill_stacked_spine,
    run_stacked_puf_pass,
    sample_acs_households,
    stacked_completeness_gate,
    stacked_gap_fill_plan,
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


def test_production_sampling_is_uniform_and_composition_preserving() -> None:
    asec = _asec_source()
    acs = _acs_source()
    result = assemble_stacked_spine(
        asec,
        acs,
        sample_fraction=0.5,
        sample_seed=578,
    )

    assert result.receipt["version"] == 2
    assert result.receipt["sample_fraction"] == 0.5
    assert result.receipt["sample_seed"] == 578
    samples = result.receipt["survey_samples"]
    assert samples["asec"]["realized_household_count"] == 1
    assert samples["acs"]["realized_household_count"] == 5
    for channel, source in (("asec", asec), ("acs", acs)):
        sample = samples[channel]
        assert sample["fraction"] == 0.5
        assert sample["seed"] == 578
        assert np.isclose(
            sample["normalized_household_mass"],
            source.weights_for("household").total,
        )

    # The rung changes row counts, not either arm's population meaning.
    assert np.isclose(
        result.frame.weights_for("household").total,
        asec.weights_for("household").total,
    )
    validate_stacked_spine_frame(result.frame, boundary="production sample fixture")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("sample_fraction", 1.0, "sample fraction does not match"),
        ("sample_seed", 579, "sample seed does not match"),
    ),
)
def test_production_sampling_identity_mutations_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    result = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        sample_fraction=0.5,
        sample_seed=578,
    )
    manifest = deepcopy(result.receipt)
    manifest[field] = value
    tampered = Frame(
        {entity: result.frame.table(entity) for entity in result.frame.entities},
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata={
            **result.frame.metadata,
            STACKED_SPINE_MANIFEST_KEY: manifest,
        },
    )

    with pytest.raises(ValueError, match=match):
        validate_stacked_spine_frame(tampered, boundary="mutated sample identity")


def test_production_and_legacy_sampling_controls_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="exactly one complete sampling control"):
        assemble_stacked_spine(
            _asec_source(),
            _acs_source(),
            acs_sample_fraction=0.5,
            acs_sample_seed=578,
            sample_fraction=0.5,
            sample_seed=578,
        )


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
            "sample fraction does not match",
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


def test_weight_harmonization_receipts_use_the_live_selected_anchor() -> None:
    result = assemble_stacked_spine(
        _asec_source(),
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
        mass_anchor_channel="acs",
    )

    assert result.frame.weights_for("household").total == 550.0
    harmonization = result.receipt["weight_harmonization"]
    for channel in ("asec", "acs"):
        assert harmonization[channel]["declared_allocation"] == 275.0
        assert harmonization[channel]["allocated_mass"] == 275.0

    metadata = {
        **result.frame.metadata,
        STACKED_SPINE_MANIFEST_KEY: deepcopy(result.receipt),
    }
    metadata[STACKED_SPINE_MANIFEST_KEY]["weight_harmonization"]["asec"][
        "declared_allocation"
    ] = 200.0
    forged = Frame(
        {entity: result.frame.table(entity) for entity in result.frame.entities},
        result.frame.schema,
        {
            entity: result.frame.weights_for(entity)
            for entity in result.frame.weighted_entities
        },
        result.frame.strata,
        mass_log=result.frame.mass_log,
        metadata=metadata,
    )
    with pytest.raises(
        ValueError,
        match=(
            "declared 'asec' allocation 200.0 differs from share 0.5 times "
            "live anchor mass 550.0"
        ),
    ):
        validate_stacked_spine_frame(forged, boundary="forged allocation")


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


def _asec_detail_source() -> Frame:
    """ASEC arm with observed tax-detail sentinels and PUF-pass predictors."""

    frame = _source_frame(
        household_ids=[11, 12],
        persons_per_household={11: 2},
        weights=[300.0, 100.0],
        stratum="asec_2024",
    )
    person = frame.table("person").copy()
    person["employment_income_before_lsr"] = np.asarray([50_000.0, 20_000.0, 35_000.0])
    person["taxable_interest_income"] = np.asarray([100.0, 0.0, 200.0])
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _cloned_stacked_fixture() -> Frame:
    stacked = assemble_stacked_spine(
        _asec_detail_source(),
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
    ).frame
    return clone_us_frame_for_puf_support(stacked)


def _finalize_fixture_predictions(
    cloned: Frame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tax_unit = cloned.table("tax_unit")
    puf_mask = tax_unit[support_clone_index_column("tax_unit")].eq(1).to_numpy()
    predictions = pd.DataFrame(
        {
            "taxable_interest_income": np.full(int(puf_mask.sum()), 100.0),
            "health_savings_account_ald": np.full(int(puf_mask.sum()), 750.0),
        },
        index=tax_unit.index[puf_mask],
    )
    donor = pd.DataFrame(
        {
            "taxable_interest_income": [100.0, 200.0, 100.0, 150.0],
            "health_savings_account_ald": [750.0, 500.0, 250.0, 1_000.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    return predictions, donor


def test_finalize_preserve_nulls_keeps_unowned_cells_null() -> None:
    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)
    before_person = cloned.table("person").copy(deep=True)

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    person = finalized.table("person")
    channel = person[support_channel_column("person")]
    clone_index = person[support_clone_index_column("person")]
    native_acs = channel.eq("acs") & clone_index.eq(0)
    native_asec = channel.eq("asec") & clone_index.eq(0)
    detail = clone_index.eq(1)
    assert person.loc[native_acs, "taxable_interest_income"].isna().all()
    pd.testing.assert_series_equal(
        person.loc[native_asec, "taxable_interest_income"],
        before_person.loc[native_asec.to_numpy(), "taxable_interest_income"],
        check_names=False,
    )
    assert person.loc[detail, "taxable_interest_income"].notna().all()

    tax_unit = finalized.table("tax_unit")
    tax_unit_clone = tax_unit[support_clone_index_column("tax_unit")]
    assert tax_unit.loc[tax_unit_clone.eq(0), "health_savings_account_ald"].isna().all()
    assert (
        tax_unit.loc[tax_unit_clone.eq(1), "health_savings_account_ald"].notna().all()
    )


def test_finalize_legacy_zero_fill_reproduces_the_audited_defect() -> None:
    """Pin the run-7 boundary: legacy finalization reads absence as zero."""

    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
    )

    person = finalized.table("person")
    channel = person[support_channel_column("person")]
    clone_index = person[support_clone_index_column("person")]
    native_acs = channel.eq("acs") & clone_index.eq(0)
    assert person.loc[native_acs, "taxable_interest_income"].eq(0.0).all()
    tax_unit = finalized.table("tax_unit")
    tax_unit_clone = tax_unit[support_clone_index_column("tax_unit")]
    assert (
        tax_unit.loc[tax_unit_clone.eq(0), "health_savings_account_ald"].eq(0.0).all()
    )


def test_finalize_legacy_zero_fill_matches_run_7_protocol_5_byte_pin() -> None:
    """Pin every legacy run-7 output byte, not only its zero-region meaning."""

    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)
    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
    )

    actual = (
        finalized.table("person")["taxable_interest_income"].to_numpy(copy=True),
        finalized.table("tax_unit")["health_savings_account_ald"].to_numpy(copy=True),
    )
    actual_bytes = pickle.dumps(actual, protocol=5)
    assert actual_bytes[:2] == b"\x80\x05"
    assert hashlib.sha256(actual_bytes).hexdigest() == (
        "f2dc86f630a41c7b0eb060d7cd630bfadda423ae26c4311403116e3e6cb7720e"
    )


def test_preserve_nulls_sparsification_never_rewrites_native_rows() -> None:
    cloned = _cloned_stacked_fixture()
    predictions, donor = _finalize_fixture_predictions(cloned)
    donor = donor.assign(
        taxable_interest_income=[100.0, 0.0, 0.0, 0.0],
    )
    before_person = cloned.table("person").copy(deep=True)

    finalized = finalize_us_puf_tax_detail_predictions(
        cloned,
        donor,
        predictions.copy(),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
    )

    person = finalized.table("person")
    clone_index = person[support_clone_index_column("person")]
    channel = person[support_channel_column("person")]
    native = clone_index.eq(0)
    native_asec = native & channel.eq("asec")
    pd.testing.assert_series_equal(
        person.loc[native_asec, "taxable_interest_income"],
        before_person.loc[native_asec.to_numpy(), "taxable_interest_income"],
        check_names=False,
    )
    assert (
        person.loc[native & channel.eq("acs"), "taxable_interest_income"].isna().all()
    )
    detail_units = (
        person.loc[clone_index.eq(1)]
        .groupby("person_tax_unit_id", sort=False)["taxable_interest_income"]
        .sum()
    )
    assert (detail_units == 0.0).any()


def test_strict_recipient_predictors_fail_closed_on_absence() -> None:
    cloned = _cloned_stacked_fixture()
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="puf_predictor_employment_income"):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_employment_income",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )

    legacy = prepare_us_puf_tax_detail_chain_inputs(
        cloned,
        donor,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
    )
    assert not legacy.recipient_features.isna().any().any()
    tax_unit = cloned.table("tax_unit")
    detail_mask = tax_unit[support_clone_index_column("tax_unit")].eq(1).to_numpy()
    acs_detail = (
        tax_unit.loc[detail_mask, support_channel_column("tax_unit")]
        .eq("acs")
        .to_numpy()
    )
    zero_filled = legacy.recipient_features.loc[
        acs_detail, "puf_predictor_employment_income"
    ]
    assert zero_filled.eq(0.0).all()


def test_strict_recipient_predictors_reject_null_filing_status_before_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing filing status is terminal before status-code conversion."""

    cloned = _cloned_stacked_fixture()
    tax_unit = cloned.table("tax_unit")
    tax_unit["filing_status_input"] = "SINGLE"
    puf_mask = tax_unit[support_clone_index_column("tax_unit")].eq(1)
    tax_unit.loc[tax_unit.index[puf_mask][0], "filing_status_input"] = np.nan
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0],
            "taxable_interest_income": [120.0, 30.0],
            "weight": [1.0, 1.0],
        }
    )

    def reject_early_coercion(_values: object) -> np.ndarray:
        raise AssertionError("filing-status coercion ran before recipient validation")

    monkeypatch.setattr(
        puf_support_module,
        "_filing_status_codes",
        reject_early_coercion,
    )
    with pytest.raises(
        ValueError,
        match=r"puf_predictor_filing_status_code.*1.*recipient rows",
    ):
        prepare_us_puf_tax_detail_chain_inputs(
            cloned,
            donor,
            predictors=("puf_predictor_filing_status_code",),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=(),
            require_complete_recipient_predictors=True,
        )


def _asec_gap_source() -> Frame:
    """ASEC arm observing survey detail plus the native donor analogs."""

    frame = _source_frame(
        household_ids=[11, 12, 13, 14],
        persons_per_household={11: 2, 13: 2},
        weights=[300.0, 100.0, 200.0, 150.0],
        stratum="asec_2024",
    )
    person = frame.table("person").copy()
    count = len(person)
    person["is_female"] = np.asarray([False, True, True, False, True, False])
    person["is_household_head"] = np.asarray([True, False, True, True, False, True])
    person["employment_income_before_lsr"] = np.linspace(10_000.0, 60_000.0, count)
    person["unemployment_compensation"] = np.asarray(
        [0.0, 1_200.0, 0.0, 3_600.0, 0.0, 2_400.0]
    )
    person["is_disabled"] = np.asarray([False, False, True, False, False, True])
    for column, base in (
        ("taxable_interest_income", 100.0),
        ("tax_exempt_interest_income", 0.0),
        ("qualified_dividend_income", 50.0),
        ("non_qualified_dividend_income", 25.0),
        ("rental_income", 0.0),
        ("estate_income", 0.0),
    ):
        person[column] = np.linspace(base, base * 2 if base else 0.0, count)
    household = frame.table("household").copy()
    household["tenure_type"] = pd.Series(
        ["OWNED_WITH_MORTGAGE", "RENTED", "OWNED_OUTRIGHT", "RENTED"],
        dtype=object,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    tables["household"] = household
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _acs_gap_source() -> Frame:
    """ACS arm observing housing plus the honest native aggregates."""

    frame = _source_frame(
        household_ids=list(range(101, 111)),
        persons_per_household={103: 2},
        weights=[float(10 * position) for position in range(1, 11)],
        stratum="acs_2024_1yr",
    )
    person = frame.table("person").copy()
    count = len(person)
    person["is_female"] = np.asarray([position % 2 == 0 for position in range(count)])
    person["is_household_head"] = ~person["person_household_id"].duplicated()
    person["employment_income_before_lsr"] = np.linspace(8_000.0, 90_000.0, count)
    person["acs_interest_dividend_rental_income"] = np.asarray(
        [0.0, 400.0, 0.0, 150.0, 900.0, 0.0, 250.0, 0.0, 3_000.0, 120.0, 60.0]
    )
    person["pre_subsidy_rent"] = np.asarray(
        [
            0.0,
            14_400.0,
            0.0,
            0.0,
            9_600.0,
            12_000.0,
            0.0,
            18_000.0,
            0.0,
            7_200.0,
            15_600.0,
        ]
    )
    household = frame.table("household").copy()
    household["tenure_type"] = pd.Series(
        [
            "OWNED_OUTRIGHT",
            "RENTED",
            "OWNED_WITH_MORTGAGE",
            "OWNED_OUTRIGHT",
            "RENTED",
            "RENTED",
            "OWNED_WITH_MORTGAGE",
            "RENTED",
            "OWNED_OUTRIGHT",
            "RENTED",
        ],
        dtype=object,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    tables["household"] = household
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


_GAP_FILL_TEST_PLAN = (
    GapFillDirection(
        name="asec_survey_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={
            "person": {
                "model_required_numeric": ("unemployment_compensation",),
                "model_required_boolean": ("is_disabled",),
            }
        },
    ),
    GapFillDirection(
        name="acs_housing_to_asec",
        recipient_channel="asec",
        donor_channel="acs",
        target_families={"person": {"housing": ("pre_subsidy_rent",)}},
    ),
)


def test_production_entrypoints_take_no_authority_parameters() -> None:
    production_entrypoints = (
        gap_fill_stacked_spine,
        stacked_completeness_gate,
        by_origin_battery,
    )
    authority_parameter_tokens = {
        "authority",
        "canonical",
        "declared",
        "metric",
        "metrics",
        "plan",
        "profile",
        "registry",
        "support",
        "surface",
    }

    for entrypoint in production_entrypoints:
        authority_parameters = {
            parameter
            for parameter in inspect.signature(entrypoint).parameters
            if authority_parameter_tokens.intersection(parameter.split("_"))
        }
        assert not authority_parameters, (
            f"{entrypoint.__name__} exposes caller-controlled authority "
            f"parameter(s): {sorted(authority_parameters)}"
        )


def test_canonical_authority_objects_are_deeply_immutable() -> None:
    plan = stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN
    surface = stacked_spine_module.CANONICAL_STACKED_DECLARED_SURFACE
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    joint_registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY
    profile = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE

    assert isinstance(plan, tuple)
    with pytest.raises(TypeError):
        plan[0].target_families["person"]["model_required_numeric"] = ()
    with pytest.raises(TypeError):
        surface["person"]["model_required_numeric"] = ()
    with pytest.raises(TypeError):
        registry[("person", "puf_tax_itemization", "taxable_interest_income", 0)] = (
            "rare_incidence"
        )
    with pytest.raises(TypeError):
        joint_registry[
            (
                "person",
                "source_operator_immigration",
                ("ssn_card_type", "immigration_status_str"),
                0,
            )
        ] = "categorical_tvd"
    with pytest.raises(FrozenInstanceError):
        profile.min_effective_support = 50


def test_canonical_metric_registry_covers_the_declared_131_target_split() -> None:
    surface = stacked_spine_module.CANONICAL_STACKED_DECLARED_SURFACE
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    surface_targets = {
        (entity, family, target, 0)
        for entity, families in surface.items()
        for family, targets in families.items()
        for target in targets
    }

    assert len(surface_targets) == 131
    assert Counter(entity for entity, _family, _target, _clone in surface_targets) == {
        "person": 114,
        "tax_unit": 9,
        "spm_unit": 8,
    }
    assert (
        len(
            {
                (entity, family)
                for entity, families in surface.items()
                for family in families
            }
        )
        == 31
    )
    assert set(registry) == surface_targets
    assert Counter(registry.values()) == {
        "monetary_sign_separated": 79,
        "boolean_incidence": 48,
        "categorical_tvd": 4,
    }
    assert (
        registry[("person", "puf_tax_itemization", "taxable_interest_income", 0)]
        == "monetary_sign_separated"
    )
    gap_targets = {
        (entity, family, target, 0)
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    assert len(gap_targets) == 118
    assert gap_targets < surface_targets
    assert not {
        "bank_account_assets",
        "bond_assets",
        "stock_assets",
    } & {target for _entity, _family, target, _clone in surface_targets}


def test_explicit_test_seams_reject_the_canonical_authority() -> None:
    authority = stacked_spine_module._production_stacked_authority()
    frame = _stacked_gap_fixture()

    with pytest.raises(ValueError, match="NON-CANONICAL test authority"):
        stacked_spine_module._gap_fill_stacked_spine_with_test_authority(
            frame,
            authority=authority,
        )
    with pytest.raises(ValueError, match="NON-CANONICAL test authority"):
        stacked_spine_module._stacked_completeness_gate_with_test_authority(
            frame,
            authority=authority,
        )
    with pytest.raises(ValueError, match="NON-CANONICAL test authority"):
        stacked_spine_module._by_origin_battery_with_test_authority(
            frame,
            authority=authority,
        )


def _surface_from_gap_fill_plan(
    plan: tuple[GapFillDirection, ...],
) -> dict[str, dict[str, tuple[str, ...]]]:
    surface: dict[str, dict[str, tuple[str, ...]]] = {}
    for direction in plan:
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                surface.setdefault(entity, {})[family] = tuple(targets)
    return surface


def _gap_fill_with_test_authority(
    frame: Frame,
    *,
    plan: tuple[GapFillDirection, ...],
    **kwargs: object,
):
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=_surface_from_gap_fill_plan(plan),
        gap_fill_plan=plan,
    )
    return stacked_spine_module._gap_fill_stacked_spine_with_test_authority(
        frame,
        authority=authority,
        **kwargs,
    )


def _completeness_with_test_authority(
    frame: Frame,
    *,
    declared_surface: dict[str, dict[str, tuple[str, ...]]],
    declared_gap_fill_plan: tuple[GapFillDirection, ...],
    absence_proofs: tuple[AbsenceProof, ...] = (),
):
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=declared_surface,
        gap_fill_plan=declared_gap_fill_plan,
    )
    return stacked_spine_module._stacked_completeness_gate_with_test_authority(
        frame,
        authority=authority,
        absence_proofs=absence_proofs,
    )


def _battery_with_test_authority(
    frame: Frame,
    *,
    registry: tuple[OriginBatterySpec, ...],
):
    surface = {
        entity: {family: tuple(targets) for family, targets in families.items()}
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_DECLARED_SURFACE.items()
        )
    }
    metrics: dict[tuple[str, str, str, int], str] = {}
    for spec in registry:
        family_targets = list(surface.setdefault(spec.entity, {}).get(spec.family, ()))
        for column, metric in spec.column_metrics.items():
            if column not in family_targets:
                family_targets.append(column)
            metrics[(spec.entity, spec.family, column, spec.clone_index)] = metric
        surface[spec.entity][spec.family] = tuple(family_targets)
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        metric_registry=metrics,
    )
    return stacked_spine_module._by_origin_battery_with_test_authority(
        frame,
        authority=authority,
    )


def _stacked_gap_fixture() -> Frame:
    return assemble_stacked_spine(
        _asec_gap_source(),
        _acs_gap_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame


def test_gap_fill_plan_covers_declared_families_exactly() -> None:
    plan = stacked_gap_fill_plan()
    assert [direction.name for direction in plan] == [
        "asec_survey_to_acs",
        "acs_housing_to_asec",
    ]
    survey, housing = plan
    assert survey.recipient_channel == "acs"
    assert survey.donor_channel == "asec"
    assert housing.recipient_channel == "asec"
    assert housing.donor_channel == "acs"
    assert set(housing.target_families) == {"person"}
    assert housing.target_families["person"] == {"housing": ("pre_subsidy_rent",)}

    declared = pool_transfer_target_families()
    recombined: dict[str, dict[str, tuple[str, ...]]] = {}
    for direction in plan:
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                recombined.setdefault(entity, {})[family] = tuple(targets)
    assert recombined == {
        entity: {family: tuple(targets) for family, targets in families.items()}
        for entity, families in declared.items()
    }


def test_gap_fill_fills_both_directions_with_authority_receipts() -> None:
    stacked = _stacked_gap_fixture()
    result = _gap_fill_with_test_authority(
        stacked,
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    channel = person[support_channel_column("person")]
    acs_rows = channel.eq("acs")
    asec_rows = channel.eq("asec")
    for column in ("unemployment_compensation", "is_disabled"):
        assert person.loc[acs_rows, column].notna().all()
    assert person.loc[asec_rows, "pre_subsidy_rent"].notna().all()

    before_person = stacked.table("person")
    for column in ("unemployment_compensation", "is_disabled"):
        pd.testing.assert_series_equal(
            person.loc[asec_rows, column],
            before_person.loc[asec_rows.to_numpy(), column],
            check_names=False,
        )
    pd.testing.assert_series_equal(
        person.loc[acs_rows, "pre_subsidy_rent"],
        before_person.loc[acs_rows.to_numpy(), "pre_subsidy_rent"],
        check_names=False,
    )

    directions = result.receipt["directions"]
    survey = directions["asec_survey_to_acs"]
    assert survey["donor_selection"] == "owner_projection_of_native_donor_rows"
    assert survey["resolved_donor_channel"] is None
    unemployment = survey["targets"][
        "person/model_required_numeric/unemployment_compensation"
    ]
    assert unemployment["authorized_null_rows"] == int(acs_rows.sum())
    assert unemployment["imputed_rows"] == int(acs_rows.sum())
    assert unemployment["residual_null_rows"] == 0
    housing = directions["acs_housing_to_asec"]
    rent = housing["targets"]["person/housing/pre_subsidy_rent"]
    assert rent["imputed_rows"] == int(asec_rows.sum())
    assert rent["residual_null_rows"] == 0

    survey_transfer = result.transfer_results["asec_survey_to_acs"]
    native_predictor_used = any(
        "__acs_transfer_interest_dividend_rental_income"
        in pattern.observed_optional_predictors
        for record in survey_transfer.imputed_inputs
        for pattern in record.patterns
    )
    assert native_predictor_used


def test_gap_fill_outcome_rejects_forged_residual_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transfer cannot receipt unmodeled rows after filling every hole."""

    transfer = stacked_spine_module.transfer_acs_inputs

    def forge_unmodeled_rows(*args: object, **kwargs: object) -> object:
        result = transfer(*args, **kwargs)
        return replace(
            result,
            imputed_inputs=tuple(
                replace(record, unmodeled_recipient_rows=1)
                if record.column == "unemployment_compensation"
                else record
                for record in result.imputed_inputs
            ),
        )

    monkeypatch.setattr(
        stacked_spine_module,
        "transfer_acs_inputs",
        forge_unmodeled_rows,
    )
    with pytest.raises(
        ValueError,
        match=(
            "residual-null equation failed: residual_null_rows=0 != unmodeled_rows=1"
        ),
    ) as error:
        _gap_fill_with_test_authority(
            _stacked_gap_fixture(),
            plan=_GAP_FILL_TEST_PLAN,
            seed=578,
            n_estimators=10,
        )
    assert (
        "activation accounting equation failed: authorized_null_rows=11 "
        "!= imputed_rows=11 + unmodeled_rows=1" in str(error.value)
    )


def test_gap_fill_rejects_signed_zero_donor_byte_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Donor identity distinguishes equal-valued IEEE-754 signed zeros."""

    stacked = _stacked_gap_fixture()
    person = stacked.table("person")
    channel_column = support_channel_column("person")
    donor_rows = person[channel_column].astype(str).eq("asec")
    donor_index = person.index[
        donor_rows & person["unemployment_compensation"].eq(0.0)
    ][0]
    person.loc[donor_index, "unemployment_compensation"] = -0.0
    assert np.signbit(person.loc[donor_index, "unemployment_compensation"])

    transfer = stacked_spine_module.transfer_acs_inputs

    def flip_donor_signed_zero(*args: object, **kwargs: object) -> object:
        result = transfer(*args, **kwargs)
        result_person = result.frame.table("person").copy(deep=True)
        assert np.signbit(result_person.loc[donor_index, "unemployment_compensation"])
        result_person.loc[donor_index, "unemployment_compensation"] = 0.0
        tables = {
            entity: result.frame.table(entity) for entity in result.frame.entities
        }
        tables["person"] = result_person
        changed = Frame(
            tables,
            result.frame.schema,
            {
                entity: result.frame.weights_for(entity)
                for entity in result.frame.weighted_entities
            },
            result.frame.strata,
            mass_log=result.frame.mass_log,
            metadata=result.frame.metadata,
        )
        return replace(result, frame=changed)

    monkeypatch.setattr(
        stacked_spine_module,
        "transfer_acs_inputs",
        flip_donor_signed_zero,
    )
    with pytest.raises(
        ValueError,
        match=(
            r"asec_survey_to_acs/person/model_required_numeric/"
            r"unemployment_compensation: donor byte identity failed.*"
            r"canonical donor payload changed"
        ),
    ):
        _gap_fill_with_test_authority(
            stacked,
            plan=(_GAP_FILL_TEST_PLAN[0],),
            seed=578,
            n_estimators=10,
        )


def test_donor_byte_identity_canonicalizes_semantic_strings() -> None:
    index = pd.Index([7, 3], name="donor_row")
    object_strings = pd.Series(
        ["RENTED", None],
        index=index,
        name="tenure_type",
        dtype=object,
    )
    canonical_strings = object_strings.astype(CANONICAL_STRING_DTYPE)

    assert stacked_spine_module._canonical_donor_series_payload(
        object_strings,
        boundary="object-string donor identity",
    ) == stacked_spine_module._canonical_donor_series_payload(
        canonical_strings,
        boundary="canonical-string donor identity",
    )

    unchanged_controls = (
        pd.Series([True, False], name="native_bool", dtype=bool),
        pd.Series([True, pd.NA], name="nullable_bool", dtype="boolean"),
        pd.Series(
            pd.Categorical(["a", None], categories=["a", "b"]),
            name="native_category",
        ),
        pd.Series([None, np.nan], name="all_null_object", dtype=object),
    )
    for control in unchanged_controls:
        assert stacked_spine_module._canonical_donor_series_payload(
            control,
            boundary=f"{control.name} donor identity before",
        ) == stacked_spine_module._canonical_donor_series_payload(
            control.copy(deep=True),
            boundary=f"{control.name} donor identity after",
        )

    mixed = pd.Series(
        ["RENTED", 1],
        name="mixed_tenure_type",
        dtype=object,
    )
    with pytest.raises(TypeError, match="semantic strings cannot mix"):
        stacked_spine_module._canonical_donor_series_payload(
            mixed,
            boundary="mixed-object donor identity",
        )


def test_donor_byte_identity_ignores_string_object_alias_topology() -> None:
    shared = "".join(["dynamically", "-", "allocated", "-", "donor"])
    separate = [
        "".join(["dynamically", "-", "allocated", "-", "donor"]) for _ in range(2)
    ]
    before = pd.Series([shared, shared], name="native_label", dtype=object)
    after = pd.Series(separate, name="native_label", dtype=object)
    assert before.iloc[0] is before.iloc[1]
    assert after.iloc[0] is not after.iloc[1]

    assert stacked_spine_module._canonical_donor_series_payload(
        before,
        boundary="aliased-string donor identity before",
    ) == stacked_spine_module._canonical_donor_series_payload(
        after,
        boundary="aliased-string donor identity after",
    )


def test_donor_byte_identity_accepts_semantic_boolean_object_scalars() -> None:
    semantic_booleans = pd.Series(
        [True, np.bool_(False), None],
        name="is_disabled",
        dtype=object,
    )

    assert stacked_spine_module._canonical_donor_series_payload(
        semantic_booleans,
        boundary="semantic-boolean donor identity before",
    ) == stacked_spine_module._canonical_donor_series_payload(
        semantic_booleans.copy(deep=True),
        boundary="semantic-boolean donor identity after",
    )


def test_gap_fill_activation_authority_fails_closed_on_donor_nulls() -> None:
    stacked = _stacked_gap_fixture()
    person = stacked.table("person").copy()
    channel = person[support_channel_column("person")]
    poke = person.index[channel.eq("asec")][1]
    person.loc[poke, "unemployment_compensation"] = np.nan
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    poked = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )

    with pytest.raises(ValueError, match="donors must observe"):
        _gap_fill_with_test_authority(poked, plan=_GAP_FILL_TEST_PLAN, seed=578)


@pytest.mark.parametrize(
    ("recipient_channel", "donor_channel", "role", "missing_channel"),
    (
        ("acx_typo", "asec", "recipient", "acx_typo"),
        ("acs", "asec_typo", "donor", "asec_typo"),
    ),
)
def test_gap_fill_activation_authority_rejects_nonlive_declared_channel(
    recipient_channel: str,
    donor_channel: str,
    role: str,
    missing_channel: str,
) -> None:
    plan = (
        GapFillDirection(
            name="nonlive_channel",
            recipient_channel=recipient_channel,
            donor_channel=donor_channel,
            target_families={"person": {"complete": ("age",)}},
        ),
    )

    with pytest.raises(
        ValueError,
        match=f"declared {role} channel {missing_channel!r} has no live rows",
    ):
        _gap_fill_with_test_authority(_stacked_gap_fixture(), plan=plan, seed=578)


def test_gap_fill_fails_closed_on_missing_target_column() -> None:
    stacked = _stacked_gap_fixture()
    plan = (
        GapFillDirection(
            name="asec_survey_to_acs",
            recipient_channel="acs",
            donor_channel="asec",
            target_families={
                "person": {"model_required_numeric": ("veterans_benefits",)}
            },
        ),
    )
    with pytest.raises(ValueError, match="veterans_benefits.*absent"):
        _gap_fill_with_test_authority(stacked, plan=plan, seed=578)


def test_gap_fill_rejects_cloned_frames() -> None:
    cloned = clone_us_frame_for_puf_support(_stacked_gap_fixture())
    with pytest.raises(ValueError, match="before clone operators"):
        _gap_fill_with_test_authority(cloned, plan=_GAP_FILL_TEST_PLAN, seed=578)


def test_gap_fill_banks_per_target_via_608_store(tmp_path) -> None:
    identity = {"pilot": "stacked-gap-fill", "seed": 578}
    banks = {
        "asec_survey_to_acs": AcsTransferTargetBankStore(
            tmp_path / "survey",
            identity=identity,
        ),
        "acs_housing_to_asec": AcsTransferTargetBankStore(
            tmp_path / "housing",
            identity=identity,
        ),
    }
    first = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
        target_banks=banks,
    )
    survey_files = sorted((tmp_path / "survey" / "targets").glob("*.h5"))
    housing_files = sorted((tmp_path / "housing" / "targets").glob("*.h5"))
    assert len(survey_files) == 2
    assert len(housing_files) == 1

    resumed_banks = {
        "asec_survey_to_acs": AcsTransferTargetBankStore(
            tmp_path / "survey",
            identity=identity,
        ),
        "acs_housing_to_asec": AcsTransferTargetBankStore(
            tmp_path / "housing",
            identity=identity,
        ),
    }
    second = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
        target_banks=resumed_banks,
    )
    for column in ("unemployment_compensation", "is_disabled", "pre_subsidy_rent"):
        pd.testing.assert_series_equal(
            first.frame.table("person")[column],
            second.frame.table("person")[column],
        )
    survey_receipt = resumed_banks["asec_survey_to_acs"].receipt()
    assert survey_receipt["targets"]


def test_clone_attachment_is_seeded_exact_and_pair_weighted() -> None:
    stacked = _stacked_gap_fixture()
    attached = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    repeated = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    changed_seed = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=579,
    )

    manifest = attached.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY]
    assert manifest["eligible_household_count"] == 14
    assert manifest["requested_household_count"] == 7
    assert manifest["realized_household_count"] == 7
    assert manifest["exact_count_rule"] == "floor(fraction * eligible)"
    assert (
        manifest["selected_household_source_ids_sha256"]
        == repeated.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY][
            "selected_household_source_ids_sha256"
        ]
    )
    assert (
        manifest["selected_household_source_ids_sha256"]
        != changed_seed.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY][
            "selected_household_source_ids_sha256"
        ]
    )

    household = attached.table("household")
    clone_index = household[support_clone_index_column("household")]
    assert int(clone_index.eq(0).sum()) == 14
    assert int(clone_index.eq(1).sum()) == 7
    weights = attached.weights_for("household").values
    assert np.isclose(
        float(weights.sum()),
        float(stacked.weights_for("household").total),
        rtol=1e-12,
    )
    source_column = household.columns[household.columns.str.endswith("_source_id")][0]
    attached_ids = set(
        household.loc[clone_index.eq(1), source_column].astype(int).tolist()
    )
    incoming = dict(
        zip(
            stacked.table("household")[source_column].astype(int).tolist(),
            stacked.weights_for("household").values.tolist(),
            strict=True,
        )
    )
    for row, weight in zip(household.itertuples(index=False), weights, strict=True):
        source_id = int(getattr(row, source_column))
        expected = (
            incoming[source_id] / 2.0
            if source_id in attached_ids
            else incoming[source_id]
        )
        assert np.isclose(weight, expected, rtol=1e-12)

    assert validate_puf_clone_attachment(attached, boundary="attachment fixture")


def test_clone_attachment_fraction_one_matches_full_clone() -> None:
    stacked = _stacked_gap_fixture()
    full = clone_us_frame_for_puf_support(stacked)
    attached = clone_us_frame_for_puf_support(
        stacked,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
    )

    assert attached.schema == full.schema
    assert attached.entities == full.entities
    assert attached.links == full.links
    assert attached.weighted_entities == full.weighted_entities
    for entity in full.entities:
        assert_frame_equal(
            attached.table(entity),
            full.table(entity),
            check_exact=True,
        )
    for link in full.links:
        assert_frame_equal(
            attached.link(link),
            full.link(link),
            check_exact=True,
        )
    for entity in full.weighted_entities:
        attached_weights = attached.weights_for(entity)
        full_weights = full.weights_for(entity)
        assert attached_weights.kind == full_weights.kind
        assert attached_weights.values.dtype == full_weights.values.dtype
        assert attached_weights.values.shape == full_weights.values.shape
        assert attached_weights.values.tobytes(
            order="C"
        ) == full_weights.values.tobytes(order="C")
    pd.testing.assert_series_equal(attached.strata, full.strata, check_exact=True)
    assert attached.mass_log == full.mass_log
    assert PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in full.metadata
    assert PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in attached.metadata
    assert attached.metadata == full.metadata
    receipt = validate_puf_clone_attachment(
        attached,
        boundary="fraction-one identity fixture",
        expected_fraction=1.0,
        expected_seed=0,
    )
    assert receipt["authority_form"] == "full_clone_identity_no_manifest"
    assert receipt["eligible_household_count"] == 14
    assert receipt["realized_household_count"] == 14
    with pytest.raises(ValueError, match="clone attachment manifest.*is absent"):
        validate_puf_clone_attachment(
            attached,
            boundary="fraction-one identity without declared expectation",
        )


def test_full_clone_identity_validation_fails_closed() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
    )
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables.update({link: attached.link(link) for link in attached.links})
    weights = {
        entity: attached.weights_for(entity) for entity in attached.weighted_entities
    }
    household_weights = attached.weights_for("household")
    tampered_values = household_weights.values.copy()
    household_clone = attached.table("household")[
        support_clone_index_column("household")
    ]
    first_detail = int(np.flatnonzero(household_clone.eq(1).to_numpy())[0])
    tampered_values[first_detail] += 1.0
    weights["household"] = Weights(tampered_values, household_weights.kind)
    tampered = Frame(
        tables,
        attached.schema,
        weights,
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )
    with pytest.raises(ValueError, match="full-clone identity failed"):
        validate_puf_clone_attachment(
            tampered,
            boundary="tampered full clone",
            expected_fraction=1.0,
            expected_seed=0,
        )

    asymmetric_metadata = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata={
            **attached.metadata,
            PUF_CLONE_ATTACHMENT_MANIFEST_KEY: {"unexpected": True},
        },
    )
    with pytest.raises(ValueError, match="full-clone metadata symmetry failed"):
        validate_puf_clone_attachment(
            asymmetric_metadata,
            boundary="metadata-asymmetric full clone",
            expected_fraction=1.0,
            expected_seed=0,
        )


def test_clone_attachment_configuration_fails_closed() -> None:
    stacked = _stacked_gap_fixture()
    with pytest.raises(ValueError, match="provided together"):
        clone_us_frame_for_puf_support(stacked, clone_attachment_fraction=0.5)
    with pytest.raises(ValueError, match="assembled frame"):
        clone_us_frame_for_puf_support(
            _asec_gap_source(),
            clone_attachment_fraction=0.5,
            clone_attachment_seed=1,
        )
    with pytest.raises(ValueError, match="floors to zero"):
        clone_us_frame_for_puf_support(
            stacked,
            clone_attachment_fraction=0.01,
            clone_attachment_seed=1,
        )


def test_clone_attachment_manifest_mutation_fails_closed() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
    )
    manifest = {
        key: value
        for key, value in attached.metadata[PUF_CLONE_ATTACHMENT_MANIFEST_KEY].items()
    }
    manifest["realized_household_count"] = 8
    manifest["requested_household_count"] = 8
    tampered = Frame(
        {entity: attached.table(entity) for entity in attached.entities},
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata={
            **attached.metadata,
            PUF_CLONE_ATTACHMENT_MANIFEST_KEY: manifest,
        },
    )
    with pytest.raises(ValueError, match="violates floor"):
        validate_puf_clone_attachment(tampered, boundary="tampered attachment")


def test_run_stacked_puf_pass_imputes_only_the_attached_arm() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0, 70_000.0, 22_000.0],
            "taxable_interest_income": [120.0, 30.0, 900.0, 0.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        gap_filled,
        donor,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
        seed=578,
        n_estimators=10,
    )

    person = result.frame.table("person")
    channel = person[support_channel_column("person")].astype(str)
    clone_index = person[support_clone_index_column("person")]
    assert person.loc[clone_index.eq(1), "taxable_interest_income"].notna().all()
    assert (
        person.loc[clone_index.eq(0) & channel.eq("acs"), "taxable_interest_income"]
        .isna()
        .all()
    )
    by_origin = result.receipt["recipient_person_rows_by_origin"]
    assert set(by_origin) == {"asec", "acs"}
    assert all(count > 0 for count in by_origin.values())
    assert result.receipt["doctrines"]["absent_cells"] == "preserve_nulls"

    with pytest.raises(ValueError, match="clone attachment"):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            result.frame,
            donor,
            clone_attachment_fraction=0.5,
            clone_attachment_seed=578,
        )


def test_run_stacked_puf_pass_fraction_one_receipts_out_of_frame_identity() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    donor = pd.DataFrame(
        {
            "employment_income": [45_000.0, 8_000.0, 70_000.0, 22_000.0],
            "taxable_interest_income": [120.0, 30.0, 900.0, 0.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        gap_filled,
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
        seed=578,
        n_estimators=10,
    )

    assert PUF_CLONE_ATTACHMENT_MANIFEST_KEY not in result.frame.metadata
    attachment = result.receipt["clone_attachment"]
    assert attachment["authority_form"] == "full_clone_identity_no_manifest"
    assert attachment["clone_attachment_fraction"] == 1.0
    assert attachment["clone_attachment_seed"] == 0
    assert attachment["eligible_household_count"] == 14
    assert attachment["realized_household_count"] == 14


def test_run_stacked_puf_pass_applies_clone_two_capital_gains_tail() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    tables = {entity: gap_filled.table(entity) for entity in gap_filled.entities}
    person = tables["person"].copy()
    for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
        person[column] = 0.0
    tables["person"] = person
    tax_unit = tables["tax_unit"].copy()
    tax_unit["filing_status_input"] = "SINGLE"
    for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS:
        tax_unit[column] = 0.0
    tables["tax_unit"] = tax_unit
    gap_filled = Frame(
        tables,
        gap_filled.schema,
        {
            entity: gap_filled.weights_for(entity)
            for entity in gap_filled.weighted_entities
        },
        gap_filled.strata,
        mass_log=gap_filled.mass_log,
        metadata=gap_filled.metadata,
    )
    donor = pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 1_000_001],
            "employment_income": [45_000.0, 8_000.0, 70_000.0],
            "taxable_interest_income": [120.0, 30.0, 900.0],
            "health_savings_account_ald": [0.0, 500.0, 1_000.0],
            "weight": [996.0, 3.0, 1.0],
            "filing_status_code": [1.0, 1.0, 1.0],
            PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN: [
                100_000.0,
                5_000_000.0,
                10_000_000.0,
            ],
            "short_term_capital_gains": [0.0, -10_000_000_000.0, 5_000_000_000.0],
            "long_term_capital_gains_before_response": [
                100_000.0,
                100_000_000_000.0,
                75_000_000_000.0,
            ],
            "long_term_capital_gains_on_collectibles": [
                0.0,
                2_000_000_000.0,
                1_000_000_000.0,
            ],
            "non_sch_d_capital_gains": [0.0, 3_000_000_000.0, 0.0],
            "unrecaptured_section_1250_gain": [
                0.0,
                4_000_000_000.0,
                250_000_000.0,
            ],
        }
    )

    result = run_stacked_puf_pass(
        gap_filled,
        donor,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        predictors=("puf_predictor_employment_income",),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        seed=578,
        n_estimators=10,
    )

    assert result.receipt["tail_status"] == "applied"
    tail = result.receipt["puf_capital_gains_tail_transfer"]
    assert tail["record_count"] == 2
    for entity in result.frame.entities:
        assert 2 in set(
            result.frame.table(entity)[support_clone_index_column(entity)].astype(int)
        )

    preservation = stacked_spine_module.assert_stacked_tail_cells_preserved(
        result.frame,
        tail,
    )
    assert preservation["passed"] is True
    assert preservation["tail_owned_cell_count"] == 10

    prepared, receipt = stacked_spine_module.prepare_stacked_tail_derivation(
        result.frame
    )
    if "schedule_d_capital_gain_distributions" in prepared.table("person"):
        clone_two = prepared.table("person")[support_clone_index_column("person")].eq(2)
        assert (
            prepared.table("person")
            .loc[clone_two, "schedule_d_capital_gain_distributions"]
            .isna()
            .all()
        )
    assert receipt["cleared_rows"] == int(
        prepared.table("person")[support_clone_index_column("person")].eq(2).sum()
    )
    stacked_spine_module.assert_stacked_tail_cells_preserved(prepared, tail)


def _completed_stacked_frame() -> Frame:
    """A stacked fixture whose declared surface is fully observed."""

    return assemble_stacked_spine(
        _asec_gap_source(),
        _acs_gap_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame


def test_completeness_gate_passes_on_filled_and_proven_surface() -> None:
    gap_filled = _gap_fill_with_test_authority(
        _stacked_gap_fixture(),
        plan=_GAP_FILL_TEST_PLAN,
        seed=578,
        n_estimators=10,
    ).frame
    surface = {
        "person": {
            "model_required_numeric": ("unemployment_compensation",),
            "model_required_boolean": ("is_disabled",),
            "housing": ("pre_subsidy_rent",),
        }
    }
    result = _completeness_with_test_authority(
        gap_filled,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
    )
    assert result.passed
    assert result.details["declared_targets"] == 3
    statuses = {
        label: receipt["status"] for label, receipt in result.details["targets"].items()
    }
    assert set(statuses.values()) == {"complete"}


def test_completeness_gate_names_a_silently_missing_family() -> None:
    """The run-7 catcher: a declared family with no columns fails by name."""

    surface = {
        "person": {
            "puf_tax_itemization": (
                "taxable_interest_income",
                "qualified_dividend_income",
            ),
        }
    }
    stacked = _stacked_gap_fixture()
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    person = tables["person"].drop(columns=["taxable_interest_income"])
    tables["person"] = person
    dropped = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )

    result = _completeness_with_test_authority(
        dropped,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
    )
    assert not result.passed
    missing = [
        failure
        for failure in result.failures
        if "puf_tax_itemization/taxable_interest_income" in failure
        and "missing" in failure
    ]
    assert missing
    unproven = [
        failure
        for failure in result.failures
        if "qualified_dividend_income" in failure and "authority proof" in failure
    ]
    assert unproven


def test_completeness_gate_requires_source_role_proofs_for_nulls() -> None:
    stacked = _stacked_gap_fixture()
    surface = {"person": {"puf_tax_itemization": ("taxable_interest_income",)}}

    unproven = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=(),
    )
    assert not unproven.passed
    assert any(
        "acs/clone_0" in failure and "authority proof" in failure
        for failure in unproven.failures
    )

    proven = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=(),
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="taxable_interest_income",
                channel="acs",
                clone_index=0,
                reason="pending asec_survey_to_acs gap-fill",
            ),
        ),
    )
    assert proven.passed
    receipt = proven.details["targets"][
        "person/puf_tax_itemization/taxable_interest_income"
    ]
    assert receipt["status"] == "proven_absent"
    assert receipt["proven"]["acs/clone_0"]["reason"] == (
        "pending asec_survey_to_acs gap-fill"
    )


def test_completeness_gate_rejects_wildcard_for_declared_donor_hole() -> None:
    stacked = _stacked_gap_fixture()
    surface = {"person": {"model_required_numeric": ("unemployment_compensation",)}}
    wildcard = AbsenceProof(
        entity="person",
        column="unemployment_compensation",
        channel="*",
        clone_index=0,
        reason="WILDCARD LAUNDER",
    )

    laundered = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
        absence_proofs=(wildcard,),
    )
    assert not laundered.passed
    assert any(
        "origin-exact authority proof" in failure
        and "asec_survey_to_acs" in failure
        and "donor 'asec'" in failure
        for failure in laundered.failures
    )

    exact = _completeness_with_test_authority(
        stacked,
        declared_surface=surface,
        declared_gap_fill_plan=_GAP_FILL_TEST_PLAN,
        absence_proofs=(replace(wildcard, channel="acs", reason="DECLARED EXACT"),),
    )
    assert exact.passed
    proof = exact.details["targets"][
        "person/model_required_numeric/unemployment_compensation"
    ]["proven"]["acs/clone_0"]
    assert proof["authority_form"] == "origin_exact_recipient"
    assert proof["declared_direction"] == "asec_survey_to_acs"
    assert proof["declared_donor_channel"] == "asec"


def test_completeness_gate_empty_plan_cannot_launder_canonical_target() -> None:
    surface = {"person": {"model_required_numeric": ("unemployment_compensation",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=(),
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        _stacked_gap_fixture(),
        authority=authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="unemployment_compensation",
                channel="*",
                clone_index=0,
                reason="EMPTY PLAN LAUNDER",
            ),
        ),
    )

    assert not result.passed
    assert any(
        "person/model_required_numeric/unemployment_compensation" in failure
        and "canonical gap-fill plan" in failure
        and "recipient 'acs'" in failure
        and "donor 'asec'" in failure
        and "wildcard authority is forbidden" in failure
        for failure in result.failures
    )


def test_completeness_gate_empty_surface_is_terminal() -> None:
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={},
        gap_fill_plan=(),
        metric_registry={},
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        _stacked_gap_fixture(),
        authority=authority,
    )

    assert not result.passed
    assert result.details["declared_targets"] == 0
    assert any(
        "declared stacked surface contains zero targets" in failure
        for failure in result.failures
    )


def test_completeness_gate_rejects_origin_exact_proof_for_declared_donor() -> None:
    stacked = _stacked_gap_fixture()
    person = stacked.table("person").copy()
    channel = person[support_channel_column("person")].astype(str)
    donor_index = person.index[channel.eq("asec")][0]
    person.loc[donor_index, "unemployment_compensation"] = np.nan
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    donor_hole = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )
    surface = {"person": {"model_required_numeric": ("unemployment_compensation",)}}
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface=surface,
        gap_fill_plan=_GAP_FILL_TEST_PLAN,
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        donor_hole,
        authority=authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="unemployment_compensation",
                channel="asec",
                clone_index=0,
                reason="DONOR-ORIGIN LAUNDER",
            ),
            AbsenceProof(
                entity="person",
                column="unemployment_compensation",
                channel="acs",
                clone_index=0,
                reason="DECLARED RECIPIENT",
            ),
        ),
    )

    assert not result.passed
    assert any(
        "person/model_required_numeric/unemployment_compensation" in failure
        and "origin-exact authority proof is valid only for declared recipient 'acs'"
        in failure
        and "declared donor 'asec'" in failure
        for failure in result.failures
    )
    target = result.details["targets"][
        "person/model_required_numeric/unemployment_compensation"
    ]
    assert "asec/clone_0" not in target["proven"]


def test_completeness_receipts_bind_live_authority_per_target() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    assert canonical.passed, canonical.failures
    assert canonical.details["declared_targets"] == 131
    authority = canonical.details["authority"]
    assert authority["authority_form"] == "CANONICAL"
    assert authority["canonical"] is True
    battery = by_origin_battery(frame)
    assert battery.passed, battery.failures
    assert battery.details["authority"] == authority
    canonical_manifest = GateReport((canonical, battery)).to_manifest()
    assert canonical_manifest["passed"] is True
    assert (
        stacked_spine_module._authority_receipt(
            stacked_spine_module._production_stacked_authority()
        )
        == authority
    )
    plan_sha256 = authority["components"]["gap_fill_plan"]["sha256"]
    surface_sha256 = authority["components"]["declared_surface"]["sha256"]
    for receipt in canonical.details["targets"].values():
        assert receipt["authority_form"] == "observed_complete"
        assert receipt["plan_sha256"] == plan_sha256
        assert receipt["surface_sha256"] == surface_sha256

    stacked = _stacked_gap_fixture()
    person = stacked.table("person").copy()
    person["test_unplanned_absence"] = np.nan
    tables = {entity: stacked.table(entity) for entity in stacked.entities}
    tables["person"] = person
    custom_frame = Frame(
        tables,
        stacked.schema,
        {entity: stacked.weights_for(entity) for entity in stacked.weighted_entities},
        stacked.strata,
        mass_log=stacked.mass_log,
        metadata=stacked.metadata,
    )
    custom_authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={"person": {"test_only": ("test_unplanned_absence",)}},
        gap_fill_plan=(),
    )
    custom = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        custom_frame,
        authority=custom_authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="test_unplanned_absence",
                channel="*",
                clone_index=0,
                reason="test-only unplanned surface",
            ),
        ),
    )
    assert custom.passed, custom.failures
    custom_top = custom.details["authority"]
    assert custom_top["authority_form"] == "NON-CANONICAL"
    custom_target = custom.details["targets"]["person/test_only/test_unplanned_absence"]
    assert custom_target["authority_form"] == "wildcard_no_declared_donor_plan"
    assert (
        custom_target["plan_sha256"]
        == custom_top["components"]["gap_fill_plan"]["sha256"]
    )
    assert (
        custom_target["surface_sha256"]
        == custom_top["components"]["declared_surface"]["sha256"]
    )
    with pytest.raises(
        ValueError,
        match="non-canonical stacked authority is forbidden",
    ):
        stacked_spine_module._validate_production_authority_receipt(
            custom_top,
            boundary="test production artifact",
        )
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((custom,)).to_manifest()
    custom.details["authority"]["production_manifest_permitted"] = True
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((custom,)).to_manifest()
    forged_receipt = deepcopy(custom_top)
    forged_receipt.update(
        {
            "authority_id": "us_stacked_spine_authority",
            "authority_form": "CANONICAL",
            "declared_authority_form": "CANONICAL",
            "canonical": True,
            "canonical_identity": True,
            "canonical_content": True,
            "integrity_valid": True,
            "digest_matches_declared": True,
            "production_manifest_permitted": True,
            "declared_sha256": forged_receipt["sha256"],
        }
    )
    for component in forged_receipt["components"].values():
        component["declared_sha256"] = component["sha256"]
        component["digest_matches_declared"] = True
    forged_result = replace(custom, details={"authority": forged_receipt})
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((forged_result,)).to_manifest()


def test_self_digested_partial_authority_cannot_forge_production_identity() -> None:
    surface = {"person": {"test_only": ("unemployment_compensation",)}}
    forged = stacked_spine_module._make_stacked_authority(
        authority_id="us_stacked_spine_authority",
        version=1,
        gap_fill_plan=(),
        declared_surface=surface,
        metric_registry={
            ("person", "test_only", "unemployment_compensation", 0): (
                "monetary_sign_separated"
            )
        },
        support_profile=(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE),
        declared_form="CANONICAL",
    )
    result = stacked_spine_module._stacked_completeness_gate_evaluate(
        _stacked_gap_fixture(),
        authority=forged,
        production=True,
        absence_proofs=(),
    )

    assert not result.passed
    assert result.details["authority"]["canonical_identity"] is False
    assert result.details["authority"]["canonical_content"] is False
    assert any(
        "canonical stacked authority identity mismatch" in failure
        for failure in result.failures
    )
    with pytest.raises(ValueError, match="production manifest emission is forbidden"):
        GateReport((result,)).to_manifest()


def test_rebound_anchor_aliases_cannot_replace_captured_canonical_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = {"person": {"test_only": ("unemployment_compensation",)}}
    forged = stacked_spine_module._make_stacked_authority(
        authority_id="us_stacked_spine_authority",
        version=1,
        gap_fill_plan=(),
        declared_surface=surface,
        metric_registry={
            ("person", "test_only", "unemployment_compensation", 0): (
                "monetary_sign_separated"
            )
        },
        support_profile=(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE),
        declared_form="CANONICAL",
    )
    rebound = {
        "_CANONICAL_STACKED_DECLARED_SURFACE_ANCHOR": forged.declared_surface,
        "_CANONICAL_STACKED_GAP_FILL_PLAN_ANCHOR": forged.gap_fill_plan,
        "_CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY_ANCHOR": forged.metric_registry,
        "_CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE_ANCHOR": (forged.support_profile),
        "_CANONICAL_STACKED_AUTHORITY_ANCHOR": forged,
        "_STACKED_DECLARED_SURFACE": forged.declared_surface,
        "_STACKED_GAP_FILL_PLAN": forged.gap_fill_plan,
        "_BATTERY_METRIC_REGISTRY": forged.metric_registry,
        "_BATTERY_SUPPORT_PROFILE": forged.support_profile,
    }
    for name, value in rebound.items():
        monkeypatch.setattr(stacked_spine_module, name, value)

    result = stacked_completeness_gate(_stacked_gap_fixture())

    assert not result.passed
    assert result.details["authority"]["canonical"] is False
    assert result.details["authority"]["canonical_identity"] is False
    assert any(
        "canonical stacked authority identity mismatch" in failure
        for failure in result.failures
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "support_threshold",
        "surface_count",
        "direction_count",
        "target_binding",
    ),
)
def test_stacked_manifest_rejects_pre_emission_nested_receipt_mutation(
    mutation: str,
) -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    result = stacked_completeness_gate(frame)
    assert result.passed, result.failures
    authority = result.details["authority"]
    if mutation == "support_threshold":
        authority["components"]["support_profile"]["min_effective_support"] = 50
    elif mutation == "surface_count":
        authority["components"]["declared_surface"]["target_count"] = 0
    elif mutation == "direction_count":
        authority["components"]["gap_fill_plan"]["direction_count"] = 0
    else:
        target = next(iter(result.details["targets"].values()))
        target["authority_form"] = "wildcard_no_declared_donor_plan"
        target["plan_sha256"] = "0" * 64

    with pytest.raises(
        ValueError,
        match="details changed after evaluation.*manifest emission is forbidden",
    ):
        GateReport((result,)).to_manifest()


@pytest.mark.parametrize("replacement", ({}, None))
def test_stacked_manifest_requires_the_authority_receipt(
    replacement: object,
) -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    details = deepcopy(dict(canonical.details))
    if replacement is None:
        details["authority"] = None
    else:
        details.pop("authority")
    missing = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="no stacked authority receipt.*manifest emission is forbidden",
    ):
        GateReport((missing,)).to_manifest()


def test_fresh_gate_result_cannot_graft_canonical_authority_onto_test_surface() -> None:
    frame = _stacked_gap_fixture()
    person = frame.table("person").copy()
    person["test_unplanned_absence"] = np.nan
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    custom_frame = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    test_authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={"person": {"test_only": ("test_unplanned_absence",)}},
        gap_fill_plan=(),
    )
    custom = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        custom_frame,
        authority=test_authority,
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="test_unplanned_absence",
                channel="*",
                clone_index=0,
                reason="test-only wildcard",
            ),
        ),
    )
    canonical = stacked_completeness_gate(
        _battery_frame(
            {
                "taxable_interest_income": (
                    np.asarray([100.0] * 8),
                    np.asarray([100.0] * 11),
                )
            }
        )
    )
    grafted_details = deepcopy(dict(custom.details))
    grafted_details["authority"] = deepcopy(canonical.details["authority"])
    grafted = replace(custom, details=grafted_details)

    with pytest.raises(
        ValueError,
        match="must declare exactly 131 targets.*manifest emission is forbidden",
    ):
        GateReport((grafted,)).to_manifest()


def test_fresh_gate_result_cannot_forge_a_donor_origin_proof() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    details = deepcopy(dict(canonical.details))
    authority = details["authority"]
    binding = {
        "authority_sha256": authority["sha256"],
        "plan_sha256": authority["components"]["gap_fill_plan"]["sha256"],
        "surface_sha256": authority["components"]["declared_surface"]["sha256"],
    }
    label = "person/puf_tax_itemization/taxable_interest_income"
    details["targets"][label] = {
        "status": "proven_absent",
        "null_rows": 1,
        "authority_form": "origin_exact_recipient",
        **binding,
        "proven": {
            "asec/clone_0": {
                "null_rows": 1,
                "reason": "DONOR-ORIGIN FORGERY",
                "authority_form": "origin_exact_recipient",
                **binding,
                "declared_direction": "asec_survey_to_acs",
                "declared_donor_channel": "asec",
                "declared_recipient_channel": "acs",
            }
        },
        "unproven": {},
    }
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="asec/clone_0 proof is not recipient-exact.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


def test_fresh_battery_result_cannot_forge_canonical_coverage_receipts() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = by_origin_battery(frame)
    details = deepcopy(dict(canonical.details))
    details["declared_target_count"] = 1
    details["registered_target_count"] = 1
    details["comparisons"] = {
        "person/test_only/fake[clone_0]": {
            "status": "tested",
            "metric": "rare_incidence",
        }
    }
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match="coverage receipt must bind all 131 targets.*emission is forbidden",
    ):
        GateReport((forged,)).to_manifest()


def test_fresh_battery_result_cannot_relabel_a_canonical_metric() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = by_origin_battery(frame)
    details = deepcopy(dict(canonical.details))
    label = "person/puf_tax_itemization/taxable_interest_income[clone_0]"
    details["comparisons"][label]["metric"] = "rare_incidence"
    forged = replace(canonical, details=details)

    with pytest.raises(
        ValueError,
        match=(
            "taxable_interest_income.*canonical metric "
            "'monetary_sign_separated'.*emission is forbidden"
        ),
    ):
        GateReport((forged,)).to_manifest()


def test_noncanonical_stacked_receipt_cannot_escape_under_a_renamed_gate() -> None:
    authority = stacked_spine_module._make_test_stacked_authority(
        declared_surface={"person": {"test_only": ("unemployment_compensation",)}},
        gap_fill_plan=(),
    )
    custom = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        _stacked_gap_fixture(),
        authority=authority,
    )
    renamed = replace(custom, name="renamed_stacked_completeness")

    with pytest.raises(
        ValueError,
        match="unrecognized gate name.*manifest emission is forbidden",
    ):
        GateReport((renamed,)).to_manifest()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("authority_form", "NON-CANONICAL"),
        ("declared_authority_form", "NON-CANONICAL"),
        ("digest_matches_declared", False),
    ),
)
def test_stripped_noncanonical_receipt_cannot_escape_under_a_renamed_gate(
    field: str,
    value: object,
) -> None:
    stripped = GateResult(
        name="renamed_stacked_completeness",
        passed=True,
        details={"authority": {field: value}},
    )

    with pytest.raises(
        ValueError,
        match="unrecognized gate name.*manifest emission is forbidden",
    ):
        GateReport((stripped,)).to_manifest()


def test_completeness_gate_wildcard_proof_covers_every_origin() -> None:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=1.0,
        clone_attachment_seed=0,
    )
    person = attached.table("person").copy()
    person["health_savings_account_ald_person_carrier"] = np.nan
    clone_mask = person[support_clone_index_column("person")].eq(1)
    person.loc[clone_mask, "health_savings_account_ald_person_carrier"] = 100.0
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables["person"] = person
    frame = Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )
    surface = {
        "person": {
            "puf_tax_itemization": ("health_savings_account_ald_person_carrier",)
        }
    }

    unproven = _completeness_with_test_authority(
        frame,
        declared_surface=surface,
        declared_gap_fill_plan=(),
    )
    assert not unproven.passed

    proven = _completeness_with_test_authority(
        frame,
        declared_surface=surface,
        declared_gap_fill_plan=(),
        absence_proofs=(
            AbsenceProof(
                entity="person",
                column="health_savings_account_ald_person_carrier",
                channel="*",
                clone_index=0,
                reason="base arm carries no PUF detail; engine default applies",
            ),
        ),
    )
    assert proven.passed
    wildcard_receipt = proven.details["targets"][
        "person/puf_tax_itemization/health_savings_account_ald_person_carrier"
    ]["proven"]["asec/clone_0"]
    assert wildcard_receipt["authority_form"] == "wildcard_no_declared_donor_plan"


def _declared_battery_metric(entity: str, family: str, target: str) -> str:
    canonical = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY.get(
        (entity, family, target, 0)
    )
    if canonical is not None:
        return canonical
    if family in {"benefit_participation", "model_required_boolean"}:
        return "boolean_incidence"
    if family == "model_required_discrete":
        return "categorical_tvd"
    return "monetary_sign_separated"


def _complete_battery_registry(
    *extras: OriginBatterySpec,
) -> tuple[OriginBatterySpec, ...]:
    metrics: dict[tuple[str, str, int], dict[str, str]] = {}
    for (
        entity,
        family,
        target,
        clone_index,
    ), metric in stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY.items():
        metrics.setdefault((entity, family, clone_index), {})[target] = metric
    for spec in extras:
        metrics.setdefault((spec.entity, spec.family, spec.clone_index), {}).update(
            spec.column_metrics
        )
    return tuple(
        OriginBatterySpec(
            entity=entity,
            family=family,
            clone_index=clone_index,
            column_metrics=column_metrics,
        )
        for (entity, family, clone_index), column_metrics in sorted(metrics.items())
    )


def _with_declared_battery_defaults(
    frame: Frame,
    *,
    preserve: frozenset[tuple[str, str]] = frozenset(),
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for (
        entity,
        _family,
        target,
        _clone_index,
    ) in stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY:
        if (entity, target) not in preserve:
            tables[entity][target] = 1.0
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _battery_frame(columns: dict[str, tuple[np.ndarray, np.ndarray]]) -> Frame:
    """A stacked frame with hand-set asec/acs person columns.

    ``columns`` maps a column name to its (asec values, acs values) pair.
    """

    first_asec, first_acs = next(iter(columns.values()))
    asec_count = len(first_asec)
    acs_count = len(first_acs)
    asec = _source_frame(
        household_ids=list(range(11, 11 + asec_count)),
        weights=[100.0] * asec_count,
        stratum="asec_2024",
    )
    acs = _source_frame(
        household_ids=list(range(101, 101 + acs_count)),
        weights=[100.0] * acs_count,
        stratum="acs_2024_1yr",
    )

    def with_columns(frame: Frame, position: int) -> Frame:
        frame = _with_declared_battery_defaults(frame)
        person = frame.table("person").copy()
        for column, values in columns.items():
            person[column] = values[position]
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = person
        return Frame(
            tables,
            US_SCHEMA,
            {"household": frame.weights_for("household")},
            frame.strata,
        )

    return assemble_stacked_spine(
        with_columns(asec, 0),
        with_columns(acs, 1),
        acs_sample_fraction=1.0,
        acs_sample_seed=0,
    ).frame


def test_battery_boolean_incidence_is_declared_not_dispatched() -> None:
    frame = _battery_frame(
        {
            "matched_flag": (
                np.asarray([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
                np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]),
            ),
            "object_backed_flag": (
                np.asarray(
                    [True, True, True, False, False, False, False, False], dtype=object
                ),
                np.asarray([True] + [False] * 10, dtype=object),
            ),
        }
    )
    registry = (
        OriginBatterySpec(
            entity="person",
            family="model_required_boolean",
            column_metrics={
                "matched_flag": "boolean_incidence",
                "object_backed_flag": "boolean_incidence",
            },
        ),
    )
    result = _battery_with_test_authority(
        frame, registry=_complete_battery_registry(*registry)
    )

    assert not result.passed
    matched = result.details["comparisons"][
        "person/model_required_boolean/matched_flag[clone_0]"
    ]
    assert matched["status"] == "tested"
    assert not any("matched_flag" in failure for failure in result.failures)
    assert any(
        "object_backed_flag" in failure and "incidence ratio" in failure
        for failure in result.failures
    )


def test_battery_joint_immigration_tvd_preserves_dependence_guarantee() -> None:
    frame = _battery_frame(
        {
            "ssn_card_type": (
                np.asarray(["A"] * 4 + ["B"] * 4, dtype=object),
                np.asarray(["A"] * 4 + ["B"] * 4, dtype=object),
            ),
            "immigration_status_str": (
                np.asarray(["X"] * 4 + ["Y"] * 4, dtype=object),
                np.asarray(["Y"] * 4 + ["X"] * 4, dtype=object),
            ),
        }
    )

    result = by_origin_battery(frame)

    assert not result.passed
    for column in ("ssn_card_type", "immigration_status_str"):
        marginal = result.details["comparisons"][
            f"person/source_operator_immigration/{column}[clone_0]"
        ]
        assert marginal["total_variation_distance"] == 0.0
    joint_label = (
        "person/source_operator_immigration/"
        "joint[ssn_card_type,immigration_status_str][clone_0]"
    )
    assert result.details["comparisons"][joint_label]["total_variation_distance"] == 1.0
    assert any(joint_label in failure for failure in result.failures)


def test_battery_rejects_registry_omitting_declared_champva_before_comparisons() -> (
    None
):
    champva = "has_champva_health_coverage_at_interview"
    frame = _battery_frame(
        {
            champva: (np.ones(8), np.zeros(11)),
            "healthy_control": (
                np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
                np.asarray([1.0] * 7 + [0.0] * 4),
            ),
        }
    )
    result = _battery_with_test_authority(
        frame,
        registry=(
            OriginBatterySpec(
                entity="person",
                family="healthy_control",
                column_metrics={"healthy_control": "boolean_incidence"},
            ),
        ),
    )

    assert not result.passed
    assert result.details["tested_comparisons"] == 0
    assert any(
        f"missing declared battery target person/model_required_boolean/{champva}"
        in failure
        for failure in result.failures
    )
    assert any(
        champva in target for target in result.details["missing_declared_targets"]
    )


def test_battery_taxable_interest_metric_cannot_be_relabelled_rare_incidence() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.arange(1.0, 9.0),
                np.arange(101.0, 112.0),
            )
        }
    )
    registry = dict(stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    target = ("person", "puf_tax_itemization", "taxable_interest_income", 0)
    registry[target] = "rare_incidence"
    authority = stacked_spine_module._make_test_stacked_authority(
        metric_registry=registry,
    )
    result = stacked_spine_module._by_origin_battery_with_test_authority(
        frame,
        authority=authority,
    )

    assert not result.passed
    assert result.details["tested_comparisons"] == 0
    metric_receipt = result.details["authority"]["components"]["metric_registry"]
    assert metric_receipt["target_count"] == 131
    assert any(
        "person/puf_tax_itemization/taxable_interest_income[clone_0]" in failure
        and "authoritative metric 'monetary_sign_separated'" in failure
        and "got 'rare_incidence'" in failure
        for failure in result.failures
    )


def test_gap_fill_plan_digest_binds_direction_and_channels() -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    canonical = stacked_completeness_gate(frame)
    canonical_sha256 = canonical.details["authority"]["components"]["gap_fill_plan"][
        "sha256"
    ]
    plan = stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN
    altered = (
        replace(
            plan[0],
            name="rerouted_gap_fill",
            recipient_channel="asec",
            donor_channel="acs",
        ),
        *plan[1:],
    )
    authority = stacked_spine_module._make_test_stacked_authority(
        gap_fill_plan=altered,
    )
    result = stacked_spine_module._stacked_completeness_gate_with_test_authority(
        frame,
        authority=authority,
    )

    assert not result.passed
    receipt = result.details["authority"]
    assert receipt["authority_form"] == "NON-CANONICAL"
    assert receipt["components"]["gap_fill_plan"]["sha256"] != canonical_sha256
    assert any("canonical gap-fill direction mismatch" in f for f in result.failures)


def test_battery_sign_separated_catches_the_one_sided_hole() -> None:
    """The run-7 signature: ample support, hollow ACS leg -> terminal."""

    frame = _battery_frame(
        {
            "sentinel_amount": (
                np.asarray([500.0, 0.0, 1_200.0, 0.0, 800.0, 0.0, 2_000.0, 650.0]),
                np.zeros(11),
            ),
        }
    )
    registry = (
        OriginBatterySpec(
            entity="person",
            family="puf_tax_itemization",
            column_metrics={"sentinel_amount": "monetary_sign_separated"},
        ),
    )
    result = _battery_with_test_authority(
        frame, registry=_complete_battery_registry(*registry)
    )

    assert not result.passed
    assert any(
        "sentinel_amount" in failure and "positive-leg incidence ratio" in failure
        for failure in result.failures
    )


def test_battery_sign_separated_passes_matching_legs() -> None:
    frame = _battery_frame(
        {
            "signed_amount": (
                np.asarray(
                    [
                        100.0,
                        -100.0,
                        200.0,
                        -200.0,
                        300.0,
                        -300.0,
                        400.0,
                        -400.0,
                        500.0,
                        -500.0,
                        600.0,
                        -600.0,
                    ]
                ),
                np.asarray(
                    [
                        105.0,
                        -105.0,
                        210.0,
                        -210.0,
                        315.0,
                        -315.0,
                        420.0,
                        -420.0,
                        525.0,
                        -525.0,
                        630.0,
                        -630.0,
                    ]
                ),
            ),
        }
    )
    registry = (
        OriginBatterySpec(
            entity="person",
            family="puf_tax_itemization",
            column_metrics={"signed_amount": "monetary_sign_separated"},
        ),
    )
    result = _battery_with_test_authority(
        frame, registry=_complete_battery_registry(*registry)
    )
    assert result.passed, result.failures
    record = result.details["comparisons"][
        "person/puf_tax_itemization/signed_amount[clone_0]"
    ]
    assert record["legs"]["positive"]["quantile_envelope_distance"] <= 0.25
    assert record["legs"]["negative"]["quantile_envelope_distance"] <= 0.25


def test_battery_support_awareness_and_dead_comparisons() -> None:
    frame = _battery_frame(
        {
            "rare_flag": (
                np.asarray([0.0] * 8),
                np.asarray([0.0] * 11),
            ),
        }
    )
    dead = _battery_with_test_authority(
        frame,
        registry=_complete_battery_registry(
            OriginBatterySpec(
                entity="person",
                family="take_up",
                column_metrics={"rare_flag": "rare_incidence"},
            ),
        ),
    )
    assert not dead.passed
    assert any("dead" in failure for failure in dead.failures)
    assert dead.details["support_profile"]["profile_id"] == (
        "us_stacked_origin_battery_support"
    )
    assert dead.details["support_profile"]["min_effective_support"] == 5
    assert len(dead.details["support_profile"]["sha256"]) == 64


def test_battery_rebound_support_profile_with_stale_digest_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _battery_frame(
        {
            "taxable_interest_income": (
                np.asarray([100.0] * 8),
                np.asarray([100.0] * 11),
            )
        }
    )
    rebound = replace(
        stacked_spine_module.CANONICAL_ORIGIN_BATTERY_SUPPORT_PROFILE,
        min_effective_support=50,
    )
    monkeypatch.setattr(stacked_spine_module, "_BATTERY_SUPPORT_PROFILE", rebound)
    result = by_origin_battery(frame)

    assert not result.passed
    assert result.details["tested_comparisons"] == 0
    assert any(
        "support profile live-content digest mismatch" in failure
        for failure in result.failures
    )
    profile = result.details["authority"]["components"]["support_profile"]
    assert profile["sha256"] == (
        "7ffd25d0bc4c7cca1a12b61171d8d433094a60fc56cf5a099564598841252af9"
    )
    assert profile["sha256"] != profile["declared_sha256"]


def test_battery_rejects_caller_controlled_support_threshold() -> None:
    with pytest.raises(TypeError, match="min_effective_support"):
        OriginBatterySpec(
            entity="person",
            family="take_up",
            column_metrics={"rare_flag": "rare_incidence"},
            min_effective_support=50,
        )


def test_battery_categorical_tvd_and_null_scope() -> None:
    frame = _battery_frame(
        {
            "category_field": (
                np.asarray(["A", "A", "A", "B", "B", "B", "A", "B"], dtype=object),
                np.asarray(
                    ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
                    dtype=object,
                ),
            ),
            "leaky_field": (
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                np.asarray([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0]),
            ),
        }
    )
    result = _battery_with_test_authority(
        frame,
        registry=_complete_battery_registry(
            OriginBatterySpec(
                entity="person",
                family="model_required_discrete",
                column_metrics={
                    "category_field": "categorical_tvd",
                    "leaky_field": "monetary_sign_separated",
                },
            )
        ),
    )
    assert not result.passed
    assert any(
        "leaky_field" in failure and "null value(s)" in failure
        for failure in result.failures
    )
    category = result.details["comparisons"][
        "person/model_required_discrete/category_field[clone_0]"
    ]
    assert category["total_variation_distance"] <= 0.25
    assert not any("category_field" in failure for failure in result.failures)


def test_battery_spec_rejects_undeclared_metric_kinds() -> None:
    with pytest.raises(ValueError, match="unknown metric kind"):
        OriginBatterySpec(
            entity="person",
            family="take_up",
            column_metrics={"anything": "dtype_dispatch"},
        )


def test_pilot_configuration_is_the_ratified_ten_percent() -> None:
    assert STACKED_PILOT_ACS_SAMPLE_FRACTION == 0.10
    assert STACKED_PILOT_ACS_SAMPLE_SEED == 578


def _asec_e2e_source() -> Frame:
    frame = _source_frame(
        household_ids=list(range(1_001, 1_041)),
        weights=[100.0] * 40,
        stratum="asec_2024",
    )
    person = frame.table("person").copy()
    count = len(person)
    index = np.arange(count)
    person["is_female"] = index % 2 == 0
    person["is_household_head"] = True
    person["employment_income_before_lsr"] = 20_000.0 + 1_500.0 * index
    person["unemployment_compensation"] = np.where(index % 4 == 0, 2_400.0, 0.0)
    # Structurally learnable from a REQUIRED predictor with a stable share
    # under any household subsample, so the gap-fill QRF reproduces the
    # incidence on the seeded ACS sample without sampling-skew noise.
    person["is_disabled"] = person["is_female"].to_numpy()
    interest = np.where(index % 2 == 0, 1_200.0 + 40.0 * index, 0.0)
    person["taxable_interest_income"] = interest
    for column in (
        "tax_exempt_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "rental_income",
        "estate_income",
    ):
        person[column] = 0.0
    household = frame.table("household").copy()
    household["tenure_type"] = pd.Series(
        ["RENTED" if position % 2 else "OWNED_WITH_MORTGAGE" for position in range(40)],
        dtype=object,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    tables["household"] = household
    tax_unit = tables["tax_unit"].copy()
    tax_unit["health_savings_account_ald"] = np.where(
        np.arange(len(tax_unit)) % 3 == 0,
        750.0,
        0.0,
    )
    tables["tax_unit"] = tax_unit
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


def _acs_e2e_source() -> Frame:
    frame = _source_frame(
        household_ids=list(range(5_001, 5_041)),
        weights=[100.0] * 40,
        stratum="acs_2024_1yr",
    )
    person = frame.table("person").copy()
    count = len(person)
    index = np.arange(count)
    person["is_female"] = index % 2 == 1
    person["is_household_head"] = True
    person["employment_income_before_lsr"] = 21_000.0 + 1_450.0 * index
    person["acs_interest_dividend_rental_income"] = np.where(
        index % 2 == 0, 1_250.0 + 42.0 * index, 0.0
    )
    person["pre_subsidy_rent"] = np.where(index % 2 == 1, 11_000.0 + 150.0 * index, 0.0)
    household = frame.table("household").copy()
    household["tenure_type"] = pd.Series(
        ["RENTED" if position % 2 else "OWNED_OUTRIGHT" for position in range(40)],
        dtype=object,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    tables["household"] = household
    tax_unit = tables["tax_unit"].copy()
    tax_unit["health_savings_account_ald"] = np.nan
    tables["tax_unit"] = tax_unit
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
        frame.strata,
    )


_E2E_GAP_FILL_PLAN = (
    GapFillDirection(
        name="asec_survey_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={
            "person": {
                "puf_tax_itemization": ("taxable_interest_income",),
                "model_required_numeric": ("unemployment_compensation",),
                "model_required_boolean": ("is_disabled",),
            },
            "tax_unit": {
                "puf_tax_itemization": ("health_savings_account_ald",),
            },
        },
    ),
    GapFillDirection(
        name="acs_housing_to_asec",
        recipient_channel="asec",
        donor_channel="acs",
        target_families={"person": {"housing": ("pre_subsidy_rent",)}},
    ),
)


def test_end_to_end_stack_gap_fill_puf_pass_gates_and_battery(tmp_path) -> None:
    """The pilot pipeline end to end: the ACS tax-detail hole is closed.

    Run 7's failure signature was a hollow ACS income surface: the ACS spine
    carried ~zero taxable-interest incidence against ASEC's 45% because the
    PUF family silently skipped.  This walks the revised architecture at
    fixture scale — stack, banked cross-origin gap-fill with native ACS
    predictors, seeded clone attachment, one doctrine-mode PUF pass, the
    completeness gate, and the terminal by-origin battery — and proves the
    sentinel is healthy on ACS-origin rows.
    """

    stacked = assemble_stacked_spine(
        _asec_e2e_source(),
        _acs_e2e_source(),
        acs_sample_fraction=0.5,
        acs_sample_seed=578,
    ).frame

    banks = {
        direction.name: AcsTransferTargetBankStore(
            tmp_path / direction.name,
            identity={"lane": "stacked-e2e", "direction": direction.name},
        )
        for direction in _E2E_GAP_FILL_PLAN
    }
    gap_filled = _gap_fill_with_test_authority(
        stacked,
        plan=_E2E_GAP_FILL_PLAN,
        seed=578,
        n_estimators=12,
        target_banks=banks,
    )

    donor = pd.DataFrame(
        {
            "employment_income": 25_000.0 + 6_000.0 * np.arange(8),
            "taxable_interest_income": [
                1_300.0,
                0.0,
                1_500.0,
                1_800.0,
                0.0,
                2_100.0,
                1_650.0,
                1_950.0,
            ],
            "health_savings_account_ald": [
                500.0,
                0.0,
                750.0,
                1_000.0,
                250.0,
                0.0,
                800.0,
                600.0,
            ],
            "weight": [1.0] * 8,
        }
    )
    passed = stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
        gap_filled.frame,
        donor,
        clone_attachment_fraction=0.5,
        clone_attachment_seed=578,
        predictors=(
            "puf_predictor_employment_income",
            "puf_predictor_taxable_interest_income",
        ),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=("health_savings_account_ald",),
        seed=578,
        n_estimators=12,
    )

    # The audit's failure mode is now a named terminal error, not a silent
    # zero-fill: without gap-fill the strict doctrine refuses the PUF pass.
    with pytest.raises(ValueError, match="puf_predictor_taxable_interest_income"):
        stacked_spine_module._run_stacked_puf_pass_without_tail_for_test(
            stacked,
            donor,
            clone_attachment_fraction=0.5,
            clone_attachment_seed=578,
            predictors=(
                "puf_predictor_employment_income",
                "puf_predictor_taxable_interest_income",
            ),
            person_outputs=("taxable_interest_income",),
            tax_unit_outputs=("health_savings_account_ald",),
            seed=578,
            n_estimators=12,
        )

    declared_surface = {
        "person": {
            "puf_tax_itemization": ("taxable_interest_income",),
            "model_required_numeric": ("unemployment_compensation",),
            "model_required_boolean": ("is_disabled",),
            "housing": ("pre_subsidy_rent",),
        },
        "tax_unit": {
            "puf_tax_itemization": ("health_savings_account_ald",),
        },
    }
    completeness = _completeness_with_test_authority(
        passed.frame,
        declared_surface=declared_surface,
        declared_gap_fill_plan=_E2E_GAP_FILL_PLAN,
    )
    assert completeness.passed, completeness.failures

    # Drop-a-family mutation: the gate names the vanished family.
    mutated_surface = {
        "person": {
            **declared_surface["person"],
            "puf_tax_itemization": (
                "taxable_interest_income",
                "salt_refund_income",
            ),
        },
        "tax_unit": declared_surface["tax_unit"],
    }
    mutated = _completeness_with_test_authority(
        passed.frame,
        declared_surface=mutated_surface,
        declared_gap_fill_plan=_E2E_GAP_FILL_PLAN,
    )
    assert not mutated.passed
    assert any(
        "salt_refund_income" in failure and "missing" in failure
        for failure in mutated.failures
    )

    registry = (
        OriginBatterySpec(
            entity="person",
            family="puf_tax_itemization",
            column_metrics={"taxable_interest_income": "monetary_sign_separated"},
        ),
        OriginBatterySpec(
            entity="person",
            family="model_required_numeric",
            column_metrics={"unemployment_compensation": "monetary_sign_separated"},
        ),
        OriginBatterySpec(
            entity="person",
            family="model_required_boolean",
            column_metrics={"is_disabled": "boolean_incidence"},
        ),
        OriginBatterySpec(
            entity="person",
            family="housing",
            column_metrics={"pre_subsidy_rent": "monetary_sign_separated"},
        ),
    )
    battery = _battery_with_test_authority(
        _with_declared_battery_defaults(
            passed.frame,
            preserve=frozenset(
                {
                    ("person", "taxable_interest_income"),
                    ("person", "unemployment_compensation"),
                    ("person", "is_disabled"),
                    ("person", "pre_subsidy_rent"),
                }
            ),
        ),
        registry=_complete_battery_registry(*registry),
    )
    assert battery.passed, battery.failures

    sentinel = battery.details["comparisons"][
        "person/puf_tax_itemization/taxable_interest_income[clone_0]"
    ]
    assert sentinel["status"] == "tested"
    positive = sentinel["legs"]["positive"]
    assert positive["acs_incidence"] > 0.0
    assert 0.8 <= positive["incidence_ratio_acs_over_asec"] <= 1.25
    detail = passed.frame.table("person")
    detail_clone = detail[support_clone_index_column("person")].eq(1)
    detail_channel = detail[support_channel_column("person")].astype(str)
    for origin in ("asec", "acs"):
        origin_detail = detail.loc[
            detail_clone & detail_channel.eq(origin),
            "taxable_interest_income",
        ]
        assert origin_detail.notna().all()
        assert (origin_detail > 0.0).any()
