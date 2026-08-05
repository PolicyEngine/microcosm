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

import populace.build.us_runtime.puf_support as puf_support_module
from populace.build.us_runtime.acs_transfer import (
    declared_acs_transfer_target_families,
)
from populace.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from populace.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_CLONE_ATTACHMENT_MANIFEST_KEY,
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

    declared = declared_acs_transfer_target_families()
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
    result = gap_fill_stacked_spine(
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
        gap_fill_stacked_spine(poked, plan=_GAP_FILL_TEST_PLAN, seed=578)


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
        gap_fill_stacked_spine(stacked, plan=plan, seed=578)


def test_gap_fill_rejects_cloned_frames() -> None:
    cloned = clone_us_frame_for_puf_support(_stacked_gap_fixture())
    with pytest.raises(ValueError, match="before clone operators"):
        gap_fill_stacked_spine(cloned, plan=_GAP_FILL_TEST_PLAN, seed=578)


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
    first = gap_fill_stacked_spine(
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
    second = gap_fill_stacked_spine(
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
    for entity in full.entities:
        assert_frame_equal(
            attached.table(entity).reset_index(drop=True),
            full.table(entity).reset_index(drop=True),
        )
    np.testing.assert_allclose(
        attached.weights_for("household").values,
        full.weights_for("household").values,
        rtol=1e-15,
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
    gap_filled = gap_fill_stacked_spine(
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
    result = run_stacked_puf_pass(
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
        run_stacked_puf_pass(
            result.frame,
            donor,
            clone_attachment_fraction=0.5,
            clone_attachment_seed=578,
        )


def _completed_stacked_frame() -> Frame:
    """A stacked fixture whose declared surface is fully observed."""

    return assemble_stacked_spine(
        _asec_gap_source(),
        _acs_gap_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame


def test_completeness_gate_passes_on_filled_and_proven_surface() -> None:
    gap_filled = gap_fill_stacked_spine(
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
    result = stacked_completeness_gate(gap_filled, declared_surface=surface)
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

    result = stacked_completeness_gate(dropped, declared_surface=surface)
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

    unproven = stacked_completeness_gate(stacked, declared_surface=surface)
    assert not unproven.passed
    assert any(
        "acs/clone_0" in failure and "authority proof" in failure
        for failure in unproven.failures
    )

    proven = stacked_completeness_gate(
        stacked,
        declared_surface=surface,
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

    unproven = stacked_completeness_gate(frame, declared_surface=surface)
    assert not unproven.passed

    proven = stacked_completeness_gate(
        frame,
        declared_surface=surface,
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


def _battery_frame(columns: dict[str, tuple[np.ndarray, np.ndarray]]) -> Frame:
    """A stacked frame with hand-set asec/acs person columns.

    ``columns`` maps a column name to its (asec values, acs values) pair;
    the asec arm has 8 persons and the acs arm 11.
    """

    asec = _source_frame(
        household_ids=list(range(11, 19)),
        weights=[100.0] * 8,
        stratum="asec_2024",
    )
    acs = _source_frame(
        household_ids=list(range(101, 112)),
        weights=[100.0] * 11,
        stratum="acs_2024_1yr",
    )

    def with_columns(frame: Frame, position: int) -> Frame:
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
    result = by_origin_battery(frame, registry=registry)

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
    result = by_origin_battery(frame, registry=registry)

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
                    [900.0, -300.0, 1_050.0, -380.0, -420.0, 1_500.0, 0.0, 700.0]
                ),
                np.asarray(
                    [
                        980.0,
                        -350.0,
                        1_100.0,
                        -330.0,
                        -400.0,
                        1_450.0,
                        820.0,
                        0.0,
                        -310.0,
                        690.0,
                        1_200.0,
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
            min_effective_support=3,
        ),
    )
    result = by_origin_battery(frame, registry=registry)
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
    dead = by_origin_battery(
        frame,
        registry=(
            OriginBatterySpec(
                entity="person",
                family="take_up",
                column_metrics={"rare_flag": "rare_incidence"},
            ),
        ),
    )
    assert not dead.passed
    assert any("dead" in failure for failure in dead.failures)

    under_supported = by_origin_battery(
        frame,
        registry=(
            OriginBatterySpec(
                entity="person",
                family="take_up",
                column_metrics={"rare_flag": "rare_incidence"},
                min_effective_support=50,
            ),
        ),
    )
    assert under_supported.passed
    assert under_supported.details["untestable_comparisons"] == [
        "person/take_up/rare_flag[clone_0]"
    ]
    record = under_supported.details["comparisons"]["person/take_up/rare_flag[clone_0]"]
    assert record["status"] == "insufficient_support"


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
    result = by_origin_battery(
        frame,
        registry=(
            OriginBatterySpec(
                entity="person",
                family="model_required_discrete",
                column_metrics={
                    "category_field": "categorical_tvd",
                    "leaky_field": "monetary_sign_separated",
                },
            ),
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
    gap_filled = gap_fill_stacked_spine(
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
    passed = run_stacked_puf_pass(
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
        run_stacked_puf_pass(
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
    completeness = stacked_completeness_gate(
        passed.frame,
        declared_surface=declared_surface,
        absence_proofs=(
            AbsenceProof(
                entity="tax_unit",
                column="health_savings_account_ald",
                channel="*",
                clone_index=0,
                reason="base arm carries no PUF detail; engine default applies",
            ),
        ),
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
    mutated = stacked_completeness_gate(
        passed.frame,
        declared_surface=mutated_surface,
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
    battery = by_origin_battery(passed.frame, registry=registry)
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
