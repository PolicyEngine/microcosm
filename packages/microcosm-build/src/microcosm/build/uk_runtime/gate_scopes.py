"""The UK spine and local gate scopes, and the scoped gate manifest.

All three lived in ``calibration_run``, the national calibration runner. The
full build's gate battery computes through them: both its phases build the
spine gate manifest (``graph_evidence.uk_spine_gate_manifest``) and classify
the local gates (``full_gates.classify_full_gate_outcomes``), and the dense
solve keys on the source preflight. So the battery's shared hash binds this
small module instead of all of ``calibration_run`` (microcosm#931), as the
target kernels hash ``ledger_surface`` (microcosm#1492), and an edit
elsewhere in the runner re-keys neither the preflight nor, in the in-graph
build, the solve keyed on it. The calibrated gates' own code still binds the
runner, whose admin totals and CGT projection artifact their gate context is
built with. ``calibration_run`` re-exports all three for its other callers,
beside the calibration, national and local diagnostic scopes it keeps.
"""

from __future__ import annotations

from microcosm.build.country_spec import GatesManifest, load_country_spec

UK_LOCAL_GATE_SCOPE = (
    "uk_local_geography_ladder_post_calibration",
    "uk_local_brma_enum_domain",
    "uk_local_area_support",
    "uk_local_target_fit",
    "uk_local_per_family_fit",
    "uk_local_weight_ratio",
    "uk_local_weight_ess",
)

UK_SPINE_GATE_SCOPE = (
    "uk_stage_was_wealth_support",
    "uk_stage_was_wealth_coherence",
    "uk_stage_was_lisa_support",
    "uk_stage_was_lisa_anchor",
    "uk_stage_high_value_home_support_mass",
    "uk_stage_property_components",
    "uk_stage_nts_bus_travel_support",
    "uk_stage_nts_bus_travel_facts",
    "uk_stage_uc_deduction_attributes",
    "uk_stage_lcfs_consumption_support",
    "uk_stage_lcfs_consumption_energy_rake",
    "uk_stage_lcfs_consumption_bus_pricing",
    "uk_stage_lcfs_consumption_road_fuel_incidence",
    "uk_stage_lcfs_consumption_road_fuel_level",
    "uk_stage_lcfs_consumption_recomposed_totals",
    "uk_stage_lcfs_consumption_excise_quantities",
    "uk_stage_lcfs_consumption_gambling_participation",
    "uk_stage_lcfs_consumption_basis",
    "uk_stage_etb_vat_support",
    "uk_stage_etb_services_support",
    "uk_stage_etb_services_support_pricing",
    "uk_stage_frs_hmrc_spine_leaves_signal",
    "uk_stage_frs_annual_income_spells",
    "uk_stage_spi_support_channel_mass",
    "uk_stage_hmrc_spi_income_spine_identity",
    "uk_stage_hmrc_spi_income_spine_hourly_pay",
    "uk_stage_spi_benefit_coherence",
    "uk_stage_uc_take_up",
    "uk_stage_pension_credit_take_up",
    "uk_stage_child_benefit_take_up",
    "uk_stage_cgt_incidence_clone_mass",
    "uk_stage_cgt_support_split_mass",
    "uk_stage_cgt_residential_split_mass",
    "uk_stage_spi_income_band_donors_support",
    "uk_stage_hmrc_cgt_gains_spine_summary",
    "uk_stage_hmrc_cgt_asset_type_spine_summary",
    "uk_stage_cgt_incidence_anchor_composition",
    "uk_stage_salary_sacrifice_realization",
    "uk_stage_student_loans_realization",
    "uk_stage_age_tail_targets",
    "uk_stage_frs_relationships_composition",
    "uk_stage_regional_home_values_ranking",
    "uk_stage_property_purchases_fit",
    "uk_stage_uc_legacy_closure_take_up_contract",
    "uk_stage_frs_person_draws_apprentices",
    # The take-up flags at their drawing stages, before calibration (María's
    # ruling of 2026-10-29, microcosm#1095).
    "uk_stage_frs_take_up_signal",
    "uk_stage_frs_person_draws_take_up_signal",
    "uk_stage_frs_household_draws_take_up_signal",
    "uk_stage_childcare_copy_take_up_signal",
    "uk_stage_isa_subscriptions_invariants",
    "uk_stage_isa_subscriptions",
    "uk_stage_company_cars_invariants",
    "uk_stage_spi_employment_shell_fit",
    "uk_stage_aps_occupation_fit",
    # Weight-independent like the BRMA enum below, and its column exists from
    # frs_relationships onward (#791).
    "uk_ons_household_type_enum_domain",
    # Weight-independent; the column exists only from hmrc_cgt_asset_type_spine
    # onward, so it is checked at the transferred boundary, and its domain is
    # declared by the stage module (microcosm#725).
    "uk_capital_gains_asset_type_enum_domain",
    # Weight-independent; spi_housing_shell is the last spine stage that writes
    # the band, so the transferred boundary checks every row's nation
    # (microcosm#1278).
    "uk_council_tax_band_domain",
    # Weight-independent, and its column exists from frs_brma onward, so the
    # spine checks it at the assembled boundary instead of the release end.
    "uk_brma_enum_domain",
)


def uk_scoped_gate_manifest(
    scope: tuple[str, ...] | frozenset[str],
    *,
    phases: tuple[str, ...],
    policy_suffix: str,
    source: GatesManifest | None = None,
) -> GatesManifest:
    """Filter the declared UK gate spec to one battery's scope.

    The one scope-filtering implementation every scoped producer shares
    (spine build, calibration seam, release cut). ``source`` lets a caller
    that already holds a loaded spec — or a hermetic test that stubs one —
    supply it; the default is the committed package spec.
    """

    if source is None:
        source = load_country_spec("uk").gates
    entries = tuple(entry for entry in source.gates if entry.id in scope)
    missing = sorted(set(scope) - {entry.id for entry in entries})
    if missing:
        raise RuntimeError(f"UK gate scope names undeclared gate id(s): {missing}.")
    return GatesManifest(
        country=source.country,
        version=source.version,
        policy=f"{source.policy}; {policy_suffix}",
        phases=phases,
        gates=entries,
    )
