from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.spec_engine.canonical import canonical_json_bytes
from microcosm.build.spec_engine.seeds import (
    LEGACY_V1_PROTOCOL,
    SeedProtocol,
    source_inventory_sha256,
    validate_seed_protocol_wire,
)
from microcosm.build.us_runtime import sipp_tips, ssi_disability_criteria
from microcosm.build.us_runtime.acs_transfer import (
    _family_seed,
    _pattern_name,
    _pattern_seed,
)
from microcosm.build.us_runtime.multispine_pool import POOL_RANDOM_SEED
from microcosm.build.us_runtime.puf_source_agi import (
    PUF_AGGREGATE_DISAGGREGATION_SEED,
)

EXPECTED_LEGACY_V1_SITES = {
    "acs_qrf_fit_draw",
    "acs_rent_archived_training_cap",
    "acs_transfer_family_seed",
    "acs_transfer_pattern_seed",
    "adult_care_weighted_prefix_assignment",
    "capital_gains_tail_random_rank",
    "child_support_training_cap",
    "childcare_training_cap",
    "disability_benefits_training_cap",
    "eitc_take_up_assignment",
    "energy_subsidy_training_cap",
    "exact_k_pcg64_selection",
    "housing_inputs_training_cap",
    "immigration_ead_students_assignment",
    "immigration_ead_workers_assignment",
    "legacy_congressional_district_assignment",
    "legacy_geography_ladder",
    "legacy_puma_ladder",
    "medicaid_take_up_assignment",
    "other_health_insurance_training_cap",
    "pregnancy_assignment",
    "primary_qrf_fit_draw",
    "prior_year_income_training_cap",
    "puf_archived_aggregate_disaggregation",
    "puf_clone_attachment",
    "puf_live_aggregate_disaggregation",
    "retirement_contributions_training_cap",
    "retirement_distributions_training_cap",
    "scf_household_source_selector",
    "scf_auto_loan_qrf_model",
    "scf_financial_asset_qrf_model",
    "scf_net_worth_qrf_model",
    "sipp_financial_asset_qrf_models",
    "sipp_financial_asset_training_cap",
    "sipp_tip_training_cap",
    "sipp_vehicle_qrf_model",
    "sipp_vehicle_training_cap",
    "snap_discretionary_exemption_assignment",
    "snap_state_take_up_assignment",
    "snap_take_up_assignment",
    "source_aca_assignment",
    "source_count_calibration",
    "source_joint_count_calibration",
    "ssi_archived_qrf_model",
    "ssi_take_up_assignment",
    "ssi_weighted_replacement_training",
    "survey_sample_acs",
    "survey_sample_asec",
    "tanf_take_up_assignment",
    "torch_calibration_reseed",
    "weeks_unemployed_training_cap",
    "wic_claim_assignment",
    "workers_compensation_training_cap",
}

# Independent audit oracle: each ledger id must retain a current source home.
# Paths are repository-relative, not copied from the protocol implementation.
AUDITED_SOURCE_BY_SITE = {
    **dict.fromkeys(
        ("survey_sample_asec", "survey_sample_acs"),
        "packages/microcosm-build/src/microcosm/build/frame_sampling.py",
    ),
    "puf_clone_attachment": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py",
    "puf_archived_aggregate_disaggregation": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_source_agi.py",
    "puf_live_aggregate_disaggregation": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_aggregate_records.py",
    **dict.fromkeys(
        ("ssi_weighted_replacement_training", "ssi_archived_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_disability_criteria.py",
    ),
    **dict.fromkeys(
        ("sipp_vehicle_training_cap", "sipp_vehicle_qrf_model"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py",
    ),
    **dict.fromkeys(
        (
            "sipp_financial_asset_training_cap",
            "sipp_financial_asset_qrf_models",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_financial_assets.py",
    ),
    "acs_rent_archived_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/housing_inputs.py",
    "sipp_tip_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/sipp_tips.py",
    **dict.fromkeys(
        (
            "scf_household_source_selector",
            "scf_financial_asset_qrf_model",
            "scf_net_worth_qrf_model",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py",
    ),
    "scf_auto_loan_qrf_model": "packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py",
    **dict.fromkeys(
        ("acs_transfer_family_seed", "acs_transfer_pattern_seed", "acs_qrf_fit_draw"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py",
    ),
    "primary_qrf_fit_draw": "packages/microcosm-fit/src/microcosm/fit/qrf.py",
    **dict.fromkeys(
        (
            "source_aca_assignment",
            "source_count_calibration",
            "source_joint_count_calibration",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/source_runtime.py",
    ),
    "snap_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/snap_take_up.py",
    "pregnancy_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/pregnancy.py",
    "wic_claim_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/wic_claim.py",
    "snap_discretionary_exemption_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/snap_discretionary_exemption.py",
    **dict.fromkeys(
        (
            "immigration_ead_workers_assignment",
            "immigration_ead_students_assignment",
        ),
        "packages/microcosm-build/src/microcosm/build/us_runtime/immigration.py",
    ),
    "ssi_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/ssi_take_up.py",
    "medicaid_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/medicaid_take_up.py",
    "snap_state_take_up_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/snap_state_take_up.py",
    **dict.fromkeys(
        ("tanf_take_up_assignment", "eitc_take_up_assignment"),
        "packages/microcosm-build/src/microcosm/build/us_runtime/take_up.py",
    ),
    "adult_care_weighted_prefix_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/adult_care.py",
    "capital_gains_tail_random_rank": "packages/microcosm-build/src/microcosm/build/us_runtime/puf_capital_gains_tail.py",
    "torch_calibration_reseed": "packages/microcosm-calibrate/src/microcosm/calibrate/solve.py",
    "exact_k_pcg64_selection": "packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py",
    "prior_year_income_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/prior_year_income.py",
    "childcare_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/childcare.py",
    "retirement_contributions_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_contributions.py",
    "disability_benefits_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/disability_benefits.py",
    "housing_inputs_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/housing_inputs.py",
    "workers_compensation_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/workers_compensation.py",
    "retirement_distributions_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py",
    "child_support_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/child_support.py",
    "energy_subsidy_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/energy_subsidy.py",
    "other_health_insurance_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/other_health_insurance.py",
    "weeks_unemployed_training_cap": "packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py",
    "legacy_geography_ladder": "packages/microcosm-build/src/microcosm/build/us_runtime/geography_ladder.py",
    "legacy_puma_ladder": "packages/microcosm-build/src/microcosm/build/us_runtime/puma_ladder.py",
    "legacy_congressional_district_assignment": "packages/microcosm-build/src/microcosm/build/us_runtime/congressional_district_geography.py",
}


def test_legacy_v1_draw_site_inventory_is_exhaustive_and_typed() -> None:
    assert {site.id for site in LEGACY_V1_PROTOCOL.sites} == EXPECTED_LEGACY_V1_SITES
    assert len(LEGACY_V1_PROTOCOL.streams) == 14

    for site in LEGACY_V1_PROTOCOL.sites:
        wire = site.to_wire()
        assert set(wire) == {
            "id",
            "stream",
            "value_source",
            "default",
            "rng_family",
            "rng_version",
            "kernel",
            "seed_material",
            "consumption_order",
            "reset_boundary",
            "draw_condition",
            "derivation",
        }
        assert wire["stream"].startswith("stream:")
        assert wire["seed_material"]
        assert wire["consumption_order"]
        assert wire["reset_boundary"]
        assert wire["derivation"]
        assert wire["rng_version"]
        assert wire["kernel"]


def test_legacy_v1_literals_match_live_constants() -> None:
    assert (
        LEGACY_V1_PROTOCOL.site("survey_sample_asec").default
        == LEGACY_V1_PROTOCOL.site("survey_sample_acs").default
        == 578
    )
    assert LEGACY_V1_PROTOCOL.site("puf_clone_attachment").default == 578
    assert POOL_RANDOM_SEED == 0
    assert (
        LEGACY_V1_PROTOCOL.site("puf_live_aggregate_disaggregation").default
        == POOL_RANDOM_SEED
    )
    assert (
        LEGACY_V1_PROTOCOL.site("puf_archived_aggregate_disaggregation").default
        == PUF_AGGREGATE_DISAGGREGATION_SEED
        == 42
    )
    assert (
        LEGACY_V1_PROTOCOL.site("ssi_weighted_replacement_training").default
        == ssi_disability_criteria._TRAINING_SAMPLE_SEED
        == 8_386_123_572_872_638_692
    )
    assert (
        LEGACY_V1_PROTOCOL.site("ssi_archived_qrf_model").default
        == ssi_disability_criteria._ARCHIVED_MODEL_SEED
        == 42
    )
    assert (
        LEGACY_V1_PROTOCOL.site("sipp_tip_training_cap").default
        == sipp_tips._TIP_TRAINING_SAMPLE_SEED
        == 5_559_651_045_748_063_828
    )


def test_protocol_digest_covers_every_draw_site_field() -> None:
    first = LEGACY_V1_PROTOCOL.sites[0]
    changed_site = replace(first, reset_boundary="mutated_reset_boundary")
    changed = SeedProtocol(
        id=LEGACY_V1_PROTOCOL.id,
        implementation_id=LEGACY_V1_PROTOCOL.implementation_id,
        kernels=LEGACY_V1_PROTOCOL.kernels,
        sites=(changed_site, *LEGACY_V1_PROTOCOL.sites[1:]),
    )

    assert changed.implementation_sha256 != LEGACY_V1_PROTOCOL.implementation_sha256


def test_f0_legacy_protocol_contains_no_block_first_draw_site() -> None:
    assert "geography_block_draw" not in LEGACY_V1_PROTOCOL.streams
    assert all("block" not in site.id for site in LEGACY_V1_PROTOCOL.sites)


def test_audited_source_manifest_is_independent_and_total() -> None:
    assert set(AUDITED_SOURCE_BY_SITE) == EXPECTED_LEGACY_V1_SITES
    assert set(AUDITED_SOURCE_BY_SITE) == {site.id for site in LEGACY_V1_PROTOCOL.sites}
    root = Path(__file__).resolve().parents[3]
    kernel_by_id = {kernel.id: kernel for kernel in LEGACY_V1_PROTOCOL.kernels}
    stochastic_anchors = (
        "default_rng",
        "SeedSequence",
        "hashlib.",
        "QRF(",
        "random_state",
        "manual_seed",
        "PCG64",
        "_stable_unit_draws",
    )
    for site_id, relative in AUDITED_SOURCE_BY_SITE.items():
        source = (root / relative).read_text(encoding="utf-8")
        assert any(anchor in source for anchor in stochastic_anchors), site_id
        module = (
            relative.split("/src/", maxsplit=1)[1].removesuffix(".py").replace("/", ".")
        )
        site = LEGACY_V1_PROTOCOL.site(site_id)
        assert module in kernel_by_id[site.kernel].source_modules, site_id


def test_exact_blake2b_salts_keys_candidates_and_absence_conditions() -> None:
    worker = LEGACY_V1_PROTOCOL.site("immigration_ead_workers_assignment")
    student = LEGACY_V1_PROTOCOL.site("immigration_ead_students_assignment")
    assert "literal_salt=immigration:ead_workers" in worker.seed_material
    assert "literal_salt=immigration:ead_students" in student.seed_material
    assert worker.consumption_order[0] == "all_person_rows_then_worker_candidate_mask"
    assert student.consumption_order[0] == "all_person_rows_then_student_candidate_mask"

    count = LEGACY_V1_PROTOCOL.site("source_count_calibration")
    joint = LEGACY_V1_PROTOCOL.site("source_joint_count_calibration")
    assert "literal_salt=calibrate:{variable}" in count.seed_material
    assert "literal_salt=joint-calibrate:{variable}" in joint.seed_material
    assert count.draw_condition == "only_when_declared_draw_column_is_absent"
    assert joint.draw_condition == "only_when_declared_draw_column_is_absent"

    vectors = (
        (0, "snap_take_up", "2024:10:2", 9_193_979_365_434_741_258),
        (0, "immigration:ead_workers", "2024:2", 3_846_788_339_087_460_008),
        (
            578,
            "aca:takes_up_aca_if_eligible",
            "99",
            6_849_034_126_297_294_090,
        ),
    )
    for seed, salt, key, expected in vectors:
        actual = int.from_bytes(
            hashlib.blake2b(
                f"{seed}:{salt}:{key}".encode(),
                digest_size=8,
            ).digest(),
            "big",
        )
        assert actual == expected


def test_adult_tail_and_live_puf_orders_are_literal() -> None:
    adult = LEGACY_V1_PROTOCOL.site("adult_care_weighted_prefix_assignment")
    assert adult.seed_material == ("build_model_seed",)
    assert adult.consumption_order == (
        "eligible_unit_ids_stable_sorted",
        "one_permutation",
        "permuted_weight_prefix_through_usage_target",
    )
    tail = LEGACY_V1_PROTOCOL.site("capital_gains_tail_random_rank")
    assert tail.seed_material == ("build_model_seed",)
    assert tail.consumption_order[0].startswith("mergesort_recipient_household")
    live_puf = LEGACY_V1_PROTOCOL.site("puf_live_aggregate_disaggregation")
    assert live_puf.consumption_order == (
        "one_shared_generator",
        "aggregate_recids_in_runtime_declared_order",
        "per_bucket_assign_s006_choice_then_donor_choice",
    )


def test_retirement_distribution_cap_declares_support_preserving_candidates() -> None:
    site = LEGACY_V1_PROTOCOL.site("retirement_distributions_training_cap")

    assert site.seed_material == ("build_model_seed", "stage_training_cap")
    assert site.rng_family == "pandas.Series.sample RandomState(MT19937)"
    assert site.consumption_order == (
        "retain_nonzero_union_then_prioritize_positive-weight_all-target-zero_positions",
        "sample_remaining_all-target-zero_positions_with_pandas_series_sample",
        "sort_selected_positions_then_ratio-calibrate_sampled-zero_weights",
    )
    assert site.draw_condition == "donor_rows_above_5000"
    assert "DEFAULT_ZERO_ATOL" in site.derivation
    assert "full_mass/sampled_mass" in site.derivation


def test_acs_family_pattern_seed_and_label_golden_vectors() -> None:
    assert _family_seed(0, entity="person", family="family_a") == 2_974_888_678
    assert _family_seed(578, entity="tax_unit", family="transfers") == 2_219_304_948
    vectors = (
        ((), 2_696_554_395, "pattern_03_e3b0c442"),
        (("age",), 2_406_131_924, "pattern_03_013f5440"),
        (
            ("age", "employment_income"),
            3_012_383_218,
            "pattern_03_1e060688",
        ),
    )
    for observed, expected_seed, expected_name in vectors:
        assert (
            _pattern_seed(
                0,
                entity="person",
                family="benefits",
                observed_optional=observed,
            )
            == expected_seed
        )
        assert _pattern_name(3, observed) == expected_name
    derivation = LEGACY_V1_PROTOCOL.site("acs_transfer_pattern_seed").derivation
    assert "pattern_" in derivation
    assert "nul_joined_ordered_optional_predictors" in derivation


def test_kernel_attestations_are_recomputed_from_logical_source_inventory() -> None:
    for kernel in LEGACY_V1_PROTOCOL.kernels:
        assert all("/" not in module for module in kernel.source_modules)
        assert source_inventory_sha256(kernel.source_modules) == kernel.source_sha256
    assert all(site.rng_version for site in LEGACY_V1_PROTOCOL.sites)
    qrf_sites = [
        site for site in LEGACY_V1_PROTOCOL.sites if site.kernel == "regime_gated_qrf"
    ]
    assert qrf_sites
    assert all("numpy==" in site.rng_version for site in qrf_sites)


def test_protocol_rejects_duplicate_sites_out_of_range_seeds_and_bad_wire_digest() -> (
    None
):
    first = LEGACY_V1_PROTOCOL.sites[0]
    with pytest.raises(ValueError, match="outside uint64"):
        replace(first, default=-1)
    with pytest.raises(ValueError, match="outside uint64"):
        replace(first, default=2**64)
    with pytest.raises(ValueError, match="not unique"):
        SeedProtocol(
            id="legacy-v1",
            implementation_id="duplicate-test",
            kernels=LEGACY_V1_PROTOCOL.kernels,
            sites=(first, first),
        )

    wire = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    wire["sites"][0]["reset_boundary"] = "mutated"
    with pytest.raises(ValueError, match="implementation_sha256"):
        validate_seed_protocol_wire(wire)
    duplicate = copy.deepcopy(LEGACY_V1_PROTOCOL.to_wire())
    duplicate["sites"].append(copy.deepcopy(duplicate["sites"][0]))
    with pytest.raises(ValueError, match="not unique"):
        validate_seed_protocol_wire(duplicate)


def test_qrf_multi_regime_chain_has_golden_shared_draw_state() -> None:
    from microcosm.fit import RegimeGatedQRF

    row_count = 18
    x = np.linspace(-1.0, 1.0, row_count)
    donor = pd.DataFrame(
        {
            "x": x,
            "zero": np.zeros(row_count),
            "positive": np.arange(1, row_count + 1, dtype=float),
            "gated": np.where(
                np.arange(row_count) % 3 == 0,
                0.0,
                np.arange(1, row_count + 1, dtype=float),
            ),
        }
    )
    fitted = RegimeGatedQRF(n_estimators=3, seed=19, max_samples_leaf=4).fit(
        donor,
        predictors=["x"],
        targets=["zero", "positive", "gated"],
        weights="none",
    )
    assert fitted.regimes() == {
        "zero": "degenerate_zero",
        "positive": "positive_only",
        "gated": "zero_inflated_positive",
    }
    pre_state = hashlib.sha256(
        canonical_json_bytes(fitted._rng.bit_generator.state)
    ).hexdigest()
    assert (
        pre_state == "3d8eec7262f958ee65bb946f583cb6fb41b2ece24a8dbf37b4fe1ee926ef634e"
    )
    drawn = fitted.predict(donor.iloc[:6][["x"]])
    assert drawn["zero"].tolist() == [0.0] * 6
    assert drawn["positive"].to_numpy() == pytest.approx(
        [1.0, 1.4124880203324373, 3.0, 4.0, 4.405932801487469, 6.0]
    )
    assert drawn["gated"].to_numpy() == pytest.approx(
        [2.0, 3.910256579708736, 0.0, 3.235027586673718, 0.0, 0.0]
    )
    post_state = hashlib.sha256(
        canonical_json_bytes(fitted._rng.bit_generator.state)
    ).hexdigest()
    assert (
        post_state == "de1c423f97685d93be539e0d8748b61d13251bcb8b04f63ee01ca37072f7ab43"
    )
