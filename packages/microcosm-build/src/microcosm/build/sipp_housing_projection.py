"""The versioned December SIPP housing source projection recipe and diagnostics.

This module is a *recipe*, not a producer. It declares exactly which lexical
columns a December housing extraction must carry, transcribes each column's
dictionary contract, binds itself to the already-frozen parent and dictionary
pins rather than restating a second authority, and turns supplied lexical rows
into housing measurements and support diagnostics.

It opens no file, authenticates no bytes, publishes nothing, fits nothing and
performs no annualization, price rebasing or population transfer. The frozen
sixteen-field producer in :mod:`microcosm.build.sipp_december_source` is
untouched: its ``FIELDS``, pins, reconciliation counts, receipt format and
producer binding are unchanged by this lane, and this module imports its
internals rather than copying its snapshot/parse/publish pipeline.

See docs/us-sipp-december-housing-projection.md.
"""

from __future__ import annotations

import importlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

from microcosm.graph.canonical import canonical_json

from . import sipp_december_source as s
from . import sipp_housing_measurements as hm
from . import sipp_housing_predictors as hpred
from . import sipp_measurements as m

RECIPE_VERSION = "sipp-december-housing-projection/3"
RECEIPT_FORMAT = "sipp_december_housing_projection_v3"

# Version 3 adds the nine transcribed predictor columns as a separate,
# after-the-fact evidence surface. It changes no measurement class, no
# assistance rule and no rung of ``EXCLUSION_ORDER``.
#
# Version 2 is the reviewed measurement contract and, unlike version 1, it did
# produce receipts: a December extraction was run under it and reviewed. Those
# receipts are not invalidated by this bump and are not reproduced by it - a
# version 3 run reads more columns and therefore writes a different measurement
# digest and a different receipt. Version 2 remains the identity under which the
# accepted measurement evidence was recorded, and this list says that it is
# superseded as a *recipe for new extractions*, not that its outputs are void.
#
# Version 1 declared that an explicit ``ERENTSUB = 2`` admitted a household
# whatever EVOUCHER carried, folded ``assistance_not_in_universe`` into the
# unknown rung, bound no implementation identity and copied the caller's
# ``selection`` label into the receipt without counting the SSUIDs behind it. No
# version 1 receipt was ever produced.
SUPERSEDED_RECIPE_VERSIONS = (
    "sipp-december-housing-projection/1",
    "sipp-december-housing-projection/2",
)

# Modules whose bytes decide what a housing receipt means. Their digests are
# recorded in the receipt, never inside ``recipe_bytes()``: this module's own
# hash cannot live in the bytes it hashes.
IMPLEMENTATION_MODULES = (
    "microcosm.build.sipp_december_source",
    "microcosm.build.sipp_housing_measurements",
    "microcosm.build.sipp_housing_predictors",
    "microcosm.build.sipp_housing_projection",
    "microcosm.build.sipp_measurements",
)

# Byte identity of ``recipe_bytes()``. A recipe change must move this pin in the
# same commit; a silent drift between the declared contract and the transcribed
# field table refuses instead of extracting under an unreviewed contract.
RECIPE_SHA256 = "29a686a37713954952fe2b5c2a3b91fa84e9746d052056db6479b3656831dca3"

# The official document these contracts were transcribed from. Its digest is not
# a second authority: ``verify_dictionary()`` requires it to equal the frozen
# producer's already-reviewed ``dictionary_sha256`` pin.
DICTIONARY = {
    "name": "2023_SIPP_Data_Dictionary.pdf",
    "url": (
        "https://www2.census.gov/programs-surveys/sipp/tech-documentation/"
        "data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf"
    ),
    "sha256": "0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f",
    "bytes": 3991831,
    "pages": 3371,
    "page_convention": "zero_based_pdf_page_P_printed_page_is_P_plus_1",
    "allocation_code_page": 7,
    "not_in_universe_note": (
        "Zero-based page 7 states that '.N'/'N' Not-in-Universe values describe "
        "internal SIPP data. The public pipe-delimited file's missing encoding is "
        "not documented there, so no numeric missing sentinel is transcribed for "
        "the housing amounts."
    ),
}

# Ordered lexical columns. The first five are the frozen producer's identity and
# weight columns, reused unchanged; ``ERESIDENCEID`` remains the residence key
# and ``SHHADID`` is deliberately absent, since a second household identity would
# need its own authority and reconciliation.
IDENTITY_FIELDS = ("SSUID", "PNUM", "MONTHCODE", "ERESIDENCEID", "WPFINWGT")
PROJECTION_FIELDS = (
    *IDENTITY_FIELDS,
    "ETENURE",
    "ATENURE",
    "TRENTMORT",
    "ARENTMORT",
    "TUTILS",
    "AUTILS",
    "ERENTSUB",
    "ARENTSUB",
    "EVOUCHER",
    "AVOUCHER",
    "EPRDEBT",
    "APRDEBT",
    "EMHDEBT",
    "AMHDEBT",
    "EEHC_TEN",
    "AEHC_TEN",
    "EEHC_RENTSUB",
    "AEHC_RENTSUB",
    "EEHC_VOUCHER",
    "AEHC_VOUCHER",
    "TRNTMT_MED",
    "TRNTMT_STD",
    "TUTILS_MED",
    "TUTILS_STD",
    # Version 3: prospective predictor columns, observed after admission and
    # never an input to it.
    *hpred.PREDICTOR_FIELDS,
)

# Why each column beyond the seven mandated fields is carried, so the projection
# width stays reviewable rather than merely convenient.
FIELD_PURPOSE = {
    "SSUID": "frozen_producer_person_and_household_identity",
    "PNUM": "frozen_producer_person_identity",
    "MONTHCODE": "frozen_producer_month_identity",
    "ERESIDENCEID": "frozen_producer_residence_key_100001_400999",
    "WPFINWGT": "frozen_producer_published_person_weight",
    "ETENURE": "mandated_household_tenure_category",
    "ATENURE": "transcribed_allocation_flag_for_ETENURE",
    "TRENTMORT": "mandated_household_December_rent_or_mortgage_amount",
    "ARENTMORT": "transcribed_allocation_flag_for_TRENTMORT",
    "TUTILS": "mandated_separate_household_December_utilities_amount",
    "AUTILS": "transcribed_allocation_flag_for_TUTILS",
    "ERENTSUB": "mandated_household_rent_reduction_indicator",
    "ARENTSUB": "transcribed_allocation_flag_for_ERENTSUB",
    "EVOUCHER": "mandated_household_voucher_indicator",
    "AVOUCHER": "transcribed_allocation_flag_for_EVOUCHER",
    "EPRDEBT": "resolves_the_owner_branch_of_the_TRENTMORT_universe",
    "APRDEBT": "transcribed_allocation_flag_for_EPRDEBT",
    "EMHDEBT": "resolves_the_mobile_home_owner_branch_of_the_TRENTMORT_universe",
    "AMHDEBT": "transcribed_allocation_flag_for_EMHDEBT",
    "EEHC_TEN": "person_month_tenure_gate_of_the_EEHC_RENTSUB_universe",
    "AEHC_TEN": "transcribed_allocation_flag_for_EEHC_TEN",
    "EEHC_RENTSUB": "mandated_person_month_rent_subsidy_indicator",
    "AEHC_RENTSUB": "transcribed_allocation_flag_for_EEHC_RENTSUB",
    "EEHC_VOUCHER": "mandated_person_month_voucher_indicator",
    "AEHC_VOUCHER": "transcribed_allocation_flag_for_EEHC_VOUCHER",
    "TRNTMT_MED": "topcode_group_marker_required_for_the_TRENTMORT_topcode_count",
    "TRNTMT_STD": "topcode_group_marker_required_for_the_TRENTMORT_topcode_count",
    "TUTILS_MED": "topcode_group_marker_required_for_the_TUTILS_topcode_count",
    "TUTILS_STD": "topcode_group_marker_required_for_the_TUTILS_topcode_count",
    **hpred.PREDICTOR_PURPOSE,
}

# Declarations that must be explicit before any charge leaves this lane. None of
# them is applied here; each is recorded as required and not performed.
PERIOD_AND_BASIS_DECLARATIONS = {
    "amount_period": "December_2022_monthly",
    "annualization": "not_applied",
    "annualization_requirement": (
        "A December month is not one twelfth of an observed year. Annualizing is "
        "a named, separately sensitivity-tested assumption about within-year rent "
        "stability, not an arithmetic step."
    ),
    "price_basis": "USD_nominal_2022_original",
    "price_basis_requirement": (
        "Rebasing to another year requires an explicit sourced rent price index. "
        "A general CPI-U deflator and the ACS ADJHSG housing-dollar factor both "
        "carry no authority over a SIPP amount."
    ),
    "population_basis": "sipp_2023_survey_2022_reference_population",
    "population_basis_requirement": (
        "SIPP's 2022 reference population is not a target year's population. "
        "Reweighting or transfer is a separate declared step."
    ),
    "asec_cohort_pairing": (
        "The ASEC housing status is an interview-time status over survey years "
        "2023-2025 and income cohorts 2022-2024. A recipient join must state the "
        "cohort it targets; a 2022 SIPP charge model is not silently paired with "
        "a 2024 ASEC cohort."
    ),
}

NON_CLAIMS = {
    "observes_landlord_subsidy_amount": False,
    "observes_full_contract_charge_for_assisted_households": False,
    "observes_program_type_or_utility_reimbursement_split": False,
    "utilities_added_into_any_rent_quantity": False,
    "publisher_household_sampling_weight_available": False,
    "fit_transfer_calibration_or_acceptance_performed": False,
    "unassisted_counterfactual_asserted": False,
    "new_source_byte_proof_claimed": False,
    "predictor_columns_participate_in_admission": False,
    "predictor_validity_is_a_fit_admission": False,
    "sub_state_geography_derived": False,
    "monthly_income_annualized_or_rebased": False,
    "age_domain_maximum_called_a_topcode": False,
    "launch_readiness_declared": False,
}


class HousingProjectionError(ValueError):
    """Sanitized refusal at the housing projection recipe boundary."""


# The dictionary's own "File:" line for each declared grain. A predictor whose
# description names a household quantity is still printed Person-month, and the
# entry reports what the page says rather than what the concept implies.
_FILE_CLASS = {
    "household": "Household",
    "person": "Person",
    "person_month": "Person-month",
    "household_concept_person_month": "Person-month",
}


def _predictor_extra(name: str) -> dict:
    """Declarations a version 3 predictor column carries beyond a v2 contract."""
    if name not in hpred.PREDICTOR_VALUE_FIELDS:
        return {}
    return {
        "dictionary_length": hpred.PREDICTOR_LENGTHS[name],
        "prospective_predictor": True,
        "household_resolution": hpred.PREDICTOR_RESOLUTION[name],
        "limitations": list(hpred.PREDICTOR_LIMITATIONS[name]),
        "participates_in_exclusion_order": False,
    }


def _amount_entry(name: str, spec: hm._Amount) -> dict:
    return {
        "name": name,
        "kind": spec.kind,
        "grain": spec.grain,
        "file_class": _FILE_CLASS[spec.grain],
        "status_flag": spec.flag,
        "concept": spec.concept,
        "basis": spec.basis,
        "unit": spec.unit,
        "minimum": spec.low,
        "maximum": spec.high,
        "documented_numeric_missing_sentinel": None,
        "out_of_domain_sentinel": spec.sentinel,
        "universe": spec.universe,
        "universe_description": spec.universe_description,
        "dictionary_zero_based_page": spec.page,
        "dictionary_printed_page": spec.page + 1,
        "purpose": FIELD_PURPOSE[name],
        **_predictor_extra(name),
    }


def _category_entry(name: str, spec: hm._Category) -> dict:
    return {
        "name": name,
        "kind": "category",
        "grain": spec.grain,
        "file_class": _FILE_CLASS[spec.grain],
        "status_flag": spec.flag,
        "concept": spec.concept,
        "codes": list(spec.codes),
        "code_labels": dict(zip(map(str, spec.codes), spec.labels, strict=True)),
        "documented_numeric_missing_sentinel": None,
        "universe": spec.universe,
        "universe_description": spec.universe_description,
        "dictionary_zero_based_page": spec.page,
        "dictionary_printed_page": spec.page + 1,
        "purpose": FIELD_PURPOSE[name],
        **_predictor_extra(name),
    }


def field_contracts() -> tuple[dict, ...]:
    """Every transcribed column contract, ordered as the projection is written."""
    entries: dict[str, dict] = {}
    for name, spec in hm._HOUSEHOLD_CATEGORIES.items():
        entries[name] = _category_entry(name, spec)
    for name, spec in hm._PERSON_MONTH_CATEGORIES.items():
        entries[name] = _category_entry(name, spec)
    for name, spec in hm._HOUSEHOLD_AMOUNTS.items():
        entries[name] = _amount_entry(name, spec)
    for name, spec in hm._PERSON_TOPCODE_AMOUNTS.items():
        entries[name] = _amount_entry(name, spec)
    for name, spec in hpred._PREDICTOR_CATEGORIES.items():
        entries[name] = _category_entry(name, spec)
    for name, spec in hpred._PREDICTOR_AMOUNTS.items():
        entries[name] = _amount_entry(name, spec)
    ordered = []
    for name in PROJECTION_FIELDS:
        if name in entries:
            ordered.append(entries[name])
        elif name in IDENTITY_FIELDS:
            ordered.append(
                {
                    "name": name,
                    "kind": "identity",
                    "purpose": FIELD_PURPOSE[name],
                    "authority": "microcosm.build.sipp_december_source",
                }
            )
        # Allocation flags are carried by their value field's contract and are
        # validated by the closed set check below rather than listed twice.
    flags = {
        entry["status_flag"]
        for entry in entries.values()
        if entry.get("status_flag") is not None
    }
    declared = {*entries, *flags, *IDENTITY_FIELDS}
    if declared != set(PROJECTION_FIELDS):
        raise HousingProjectionError("HOUSING_RECIPE_FIELD_SET")
    return tuple(ordered)


def prospective_predictors() -> dict:
    """What the nine version 3 columns are, and everything they are not yet.

    The block exists so a later fit lane inherits the published semantics rather
    than a summary of them. Every conflict between a column's actual dictionary
    contract and a use proposed for it elsewhere is recorded here as a
    limitation on the column; none of them is resolved by choosing a value.
    """
    return {
        "purpose": (
            "Carry the source observations a later unassisted contract-rent fit "
            "would condition on, at the grain and in the period the dictionary "
            "publishes them."
        ),
        "fields": list(hpred.PREDICTOR_FIELDS),
        "rule_id": hpred.PREDICTOR_RULE_ID,
        "computed_after_admission": True,
        "exclusion_order_unchanged": True,
        "exclusion_order_exclusive": list(hm.EXCLUSION_ORDER),
        "predictor_states_added_to_the_exclusion_order": [],
        "status_order_exclusive": list(hpred.PREDICTOR_STATUS_ORDER),
        "status_is_prospective_validity_not_admission": True,
        "state_geography": {
            "field": "TEHC_ST",
            "published_code_count": 53,
            "codes_that_are_not_a_state_or_dc": list(hpred.NON_STATE_AREA_CODES),
            "state_and_dc_code_count": 51,
            "domain_is_a_published_code_set_not_the_range_01_56": True,
            "sub_state_geography_published_on_this_column": False,
            "sub_state_values_derived": False,
            "geography_is_source_geography": True,
        },
        "household_size": {
            "publisher_field": "RHNUMPER",
            "roster_statistic": "december_member_count_of_the_supplied_residence",
            "both_retained": True,
            "publisher_corrected_against_roster": False,
            "roster_corrected_against_publisher": False,
            "type_2_person_treatment_adjudicated": False,
            "disagreement_reported_as": "household_size_agreement",
        },
        "children_under_18": {
            "publisher_field": "RHNUMU18",
            "roster_statistic": (
                "count of December members whose admissible TAGE_EHC is under 18"
            ),
            "roster_statistic_requires_complete_age_coverage": True,
            "child_age_boundary": hpred.CHILD_AGE_BOUNDARY,
            "disagreement_reported_as": "children_under_18_agreement",
        },
        "income": {
            "field": "THTOTINC",
            "period": "December_2022_monthly",
            "signed": True,
            "minimum": -10_000_000_000,
            "maximum": 10_000_000_000,
            "negative_values_preserved": True,
            "annualization": "not_applied",
            "price_rebasing": "not_applied",
            "log_or_floor_transform": "not_applied",
            "period_consistent_alternative_not_carried": "THCYINCPOV",
        },
        "age": {
            "field": "TAGE_EHC",
            "status_flag": None,
            "allocation_quality_available": False,
            "domain_maximum": 90,
            "domain_maximum_called_a_topcode": False,
            "household_statistic": (
                "maximum over admissible December person observations only"
            ),
            "coverage_states": list(hpred.AGE_COVERAGE_STATES),
            "partial_coverage_is_a_lower_bound_not_a_maximum": True,
        },
        "fields_deliberately_absent": dict(hpred.PREDICTOR_FIELDS_DELIBERATELY_ABSENT),
        "shared_with_frozen_producer": {
            "field": hpred.SHARED_WITH_FROZEN_PRODUCER,
            "frozen_producer_reading": dict(hpred.FROZEN_PRODUCER_AGE_READING),
            "read_identically_here": True,
            "frozen_producer_fields_mutated": False,
            "second_contract_for_the_shared_column_introduced": False,
        },
        "not_performed": [
            "price_bridge",
            "annualization",
            "fit",
            "draw",
            "donor_mask_selection",
            "model_input_binding",
            "recipient_mapping",
            "policyengine_calculation",
        ],
    }


def recipe_document() -> dict:
    """The full recipe as a fresh canonical value; callers cannot mutate it."""
    return {
        "version": RECIPE_VERSION,
        "receipt_format": RECEIPT_FORMAT,
        "rule_id": hm.HOUSING_RULE_ID,
        "measurement_rule_id": m.MEASUREMENT_RULE_ID,
        "parent": {
            "name": s._SOURCE_NAME,
            "sha256": s._PINS.sha256,
            "bytes": s._PINS.size,
            "revision": s._PINS.revision,
            "master_sha256": s._PINS.metadata_sha256,
            "registry_sha256": s._PINS.registry_sha256,
            "dictionary_sha256": s._PINS.dictionary_sha256,
        },
        "dictionary": dict(DICTIONARY),
        "source_years": {"survey": 2023, "reference": 2022},
        "fields": list(PROJECTION_FIELDS),
        "field_contracts": list(field_contracts()),
        "residence_key": {
            "field": "ERESIDENCEID",
            "minimum": 100001,
            "maximum": 400999,
            "shhadid_introduced": False,
        },
        "grain": {
            "household_copied": list(hm.HOUSEHOLD_CATEGORIES)
            + list(hm.HOUSEHOLD_AMOUNTS),
            "person_month": list(hm.PERSON_MONTH_CATEGORIES),
            "person": list(hm.PERSON_TOPCODE_AMOUNTS),
            "predictor_household_resolved": list(hpred.HOUSEHOLD_RESOLVED_FIELDS),
            "predictor_person_resolved": list(hpred.PERSON_RESOLVED_FIELDS),
        },
        "superseded_versions": list(SUPERSEDED_RECIPE_VERSIONS),
        "prospective_predictors": prospective_predictors(),
        "quality_vocabulary": {str(k): v for k, v in m._QUALITY.items()},
        "rent_universe": {
            "field": "TRENTMORT",
            "admitted_tenure_codes": [hm.TENURE_RENTED],
            "owner_amount": "mortgage_excluded",
            "no_payment_tenure": "outside_the_published_TRENTMORT_universe",
            "owner_is_not_a_zero_rent_renter": True,
        },
        "assistance_universe": {
            "household_indicators": ["ERENTSUB", "EVOUCHER"],
            "evoucher_applies_only_when": "ERENTSUB = 1",
            "explicit_unassisted_requires": (
                "ERENTSUB = 2 with allocation flag 1 (reported) and no household "
                "copy conflict, and EVOUCHER evidence that is blank under "
                "allocation flag 0 - the only state its published universe "
                "predicts for an unassisted household"
            ),
            "voucher_evidence_states": list(hm.VOUCHER_EVIDENCE_STATES),
            "voucher_evidence_admitted_with_explicit_negative": (
                "expected_blank_not_in_universe"
            ),
            "voucher_blank_under_flag_0_is_expected_not_missing": True,
            "voucher_value_under_flag_0_distinguished_from_blank": True,
            "niu_missing_allocated_or_conflicting_recoded_as_unassisted": False,
            "person_month_indicators_determine_household_class": False,
        },
        "topcode_markers": {
            "fields": list(hm.PERSON_TOPCODE_AMOUNTS),
            "threshold_inferred": False,
            "trntmt_minimum": 1,
            "tutils_minimum": 0,
            "zero_marker_is_not_a_cleared_non_topcode": (
                "TUTILS_MED/TUTILS_STD have a published minimum of $0, so a "
                "lexical 0 is an in-domain marker value for them while the same "
                "token is invalid_domain for the $1-minimum TRNTMT_* pair. The "
                "dictionary does not say what the public file writes for a "
                "person outside a topcoded group, and no data row has been read, "
                "so the zero, invalid-domain, blank and absent marker states are "
                "counted separately and a marker total must not be interpreted "
                "before those counts are read."
            ),
        },
        "selection_authentication": {
            "modes": list(s._MODES),
            "first_32_ssuids_maximum": 32,
            "established_by": "authenticated_runner_over_the_full_december_roster",
            "recorded_in_receipt_as": "caller_asserted",
            "count_is_not_proof_of_a_relative_selection": (
                "project() sees only the rows it is handed. A selected SSUID "
                "count of 32 or fewer is consistent with, but is not proof of, "
                "those SSUIDs being the first 32 of the whole source: only a "
                "runner that enumerated the full authenticated December roster "
                "establishes that relative claim."
            ),
        },
        "implementation_binding": {
            "modules": list(IMPLEMENTATION_MODULES),
            "recorded_in": "receipt",
            "inside_recipe_bytes": False,
            "reason": (
                "A module hash placed inside the bytes it hashes cannot be "
                "computed. The receipt therefore carries the digests, so a "
                "change to a classification rule moves the receipt even when the "
                "recipe pin does not."
            ),
        },
        "exclusion_order_exclusive": list(hm.EXCLUSION_ORDER),
        "weight": {
            "person_field": "WPFINWGT",
            "household_statistic": "mean_member_WPFINWGT",
            "basis": "December_2022_mean_all_members",
            "is_publisher_household_sampling_weight": False,
            "is_a_rent_model_sampling_weight": False,
            "zero_and_unknown_weights_dropped_silently": False,
        },
        "period_and_basis_declarations": dict(PERIOD_AND_BASIS_DECLARATIONS),
        "non_claims": dict(NON_CLAIMS),
        "selection_modes": list(s._MODES),
    }


def recipe_bytes() -> bytes:
    """Canonical recipe bytes; ``RECIPE_SHA256`` pins this exact serialization."""
    return canonical_json(recipe_document())


def recipe_digest() -> str:
    payload = recipe_bytes()
    return s._key(RECIPE_VERSION, {"sha256": s._sha(payload), "bytes": len(payload)})


def verify_recipe() -> str:
    """Refuse an unreviewed contract drift, then return the recipe digest."""
    payload = recipe_bytes()
    if s._sha(payload) != RECIPE_SHA256:
        raise HousingProjectionError("HOUSING_RECIPE_TAMPERED")
    verify_dictionary()
    return recipe_digest()


def module_digests(names: Sequence[str] = IMPLEMENTATION_MODULES) -> dict[str, str]:
    """SHA-256 of each named module's source file, read at call time.

    Nothing here is circular: the digests are computed when a receipt is written
    and stored in that receipt, so this module's own bytes can be bound without
    a constant inside them ever having to equal their own hash.
    """
    if isinstance(names, str) or not all(type(n) is str and n for n in names):
        raise HousingProjectionError("HOUSING_IMPLEMENTATION_MODULES")
    return {
        name: s._sha(Path(importlib.import_module(name).__file__).read_bytes())
        for name in sorted(names)
    }


def implementation_binding() -> dict:
    """Bind the three housing modules and the two frozen modules they reuse.

    ``RECIPE_SHA256`` pins the declared contract only. A change to
    ``_assistance_class``, ``_exclusion`` or ``measurement_class`` can leave the
    recipe bytes untouched while changing what a receipt means, so the module
    digests are recorded here as well. The frozen modules are read, never
    written: their bytes and the producer binding that already-published
    vehicle-equity receipts carry are unchanged by this lane.
    """
    digests = module_digests()
    return {
        "modules": list(IMPLEMENTATION_MODULES),
        "module_sha256": digests,
        # Version 2 of the binding domain: the bound module set gained
        # ``sipp_housing_predictors`` with recipe version 3, so the payload's
        # shape changed and not only its values.
        "binding": s._key("sipp-december-housing-implementation-binding/2", digests),
        "frozen_modules_mutated": False,
        "inside_recipe_bytes": False,
        "dependency_versions_bound": False,
        "dependency_note": (
            "The measurement path is standard-library Decimal arithmetic under a "
            "fixed context; no third-party numeric library participates, so no "
            "dependency version is bound here. The frozen producer's own "
            "``_contract()`` binding, which does record dependency versions, is "
            "unchanged."
        ),
    }


def source_identity(source: m.SippSource) -> dict:
    """The source the caller asserts these rows came from, checked against pins.

    Separate from the caller's selection assertion, and never a byte proof: no
    parent byte is opened here and the supplied rows are not hashed.
    """
    verify_source(source)
    return {
        "name": s._SOURCE_NAME,
        "data_sha256": source.data_sha256,
        "metadata_sha256": source.metadata_sha256,
        "survey_year": source.survey_year,
        "reference_year": source.reference_year,
        "matches_frozen_producer_pins": True,
        "supplied_rows_verified_against_these_bytes": False,
        "new_source_byte_proof_claimed": False,
    }


def selection_evidence(
    households: Sequence[hm.HousingHousehold], selection: str
) -> dict:
    """Count the SSUIDs actually supplied and label the mode as caller-asserted.

    ``first_32_ssuids`` refuses more than 32. Thirty-two or fewer is *not* proof
    that these are the first 32 SSUIDs of the whole source, and this block says
    so: only a runner that enumerated the full authenticated December roster
    establishes a relative selection.
    """
    ssuids = tuple(sorted({h.key[0] for h in households}))
    if selection == "first_32_ssuids" and len(ssuids) > 32:
        raise HousingProjectionError("SELECTION_EXCEEDS_FIRST_32")
    return {
        "mode": selection,
        "selected_ssuid_count": len(ssuids),
        "selected_ssuid_digest": s._key(
            "sipp-december-housing-selected-ssuids/1", list(ssuids)
        ),
        "source_authentication": "caller_asserted",
        "relative_selection_established_by": (
            "authenticated_runner_over_the_full_december_roster"
        ),
        "count_is_not_proof_of_a_relative_selection": True,
        "first_32_ssuids_maximum": 32,
    }


def verify_dictionary() -> None:
    """The transcription document must be the already-pinned dictionary."""
    if DICTIONARY["sha256"] != s._PINS.dictionary_sha256:
        raise HousingProjectionError("HOUSING_RECIPE_DICTIONARY_MISMATCH")


def verify_source(source: m.SippSource) -> None:
    """Refuse any source identity that is not the frozen authenticated parent."""
    if not isinstance(source, m.SippSource):
        raise HousingProjectionError("HOUSING_SOURCE_TYPE")
    if (
        source.data_sha256 != s._PINS.sha256
        or source.metadata_sha256 != s._PINS.metadata_sha256
        or source.survey_year != 2023
        or source.reference_year != 2022
    ):
        raise HousingProjectionError("HOUSING_SOURCE_PINS_MISMATCH")


def header_positions(header: Sequence[str]) -> tuple[int, ...]:
    """Map the recipe's columns onto a supplied parent header, or refuse."""
    if not isinstance(header, Sequence) or isinstance(header, str | bytes):
        raise HousingProjectionError("HOUSING_PROJECTION_HEADER")
    names = tuple(header)
    if (
        any(type(f) is not str or not f for f in names)
        or len(set(names)) != len(names)
        or not set(PROJECTION_FIELDS) <= set(names)
    ):
        raise HousingProjectionError("HOUSING_PROJECTION_HEADER")
    return tuple(names.index(f) for f in PROJECTION_FIELDS)


@dataclass(frozen=True)
class HousingProjectionResult:
    """Housing evidence plus its aggregate receipt; no fit admission is implied."""

    households: tuple[hm.HousingHousehold, ...]
    diagnostics: Mapping[str, object]
    receipt_json: bytes
    receipt_sha256: str
    recipe_digest: str
    predictors: tuple[hpred.HouseholdPredictors, ...] = ()


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _code_counter(measures: Iterable[m.Measurement]) -> dict[str, int]:
    labels = []
    for measure in measures:
        state = hm.measurement_class(measure)
        labels.append(str(int(measure.value)) if measure.value is not None else state)
    return _counter(labels)


def _weighted_by(
    households: Sequence[hm.HousingHousehold],
    label: Callable[[hm.HousingHousehold], str],
) -> dict[str, str]:
    """Sum the household mean member weight by class, never dropping an unknown."""
    totals: dict[str, Decimal] = {}
    with localcontext(m._ARITHMETIC_CONTEXT):
        for household in households:
            if household.weight.value is None:
                key, value = "unknown_household_weight", Decimal(0)
            else:
                key, value = label(household), household.weight.value
            totals[key] = totals.get(key, Decimal(0)) + value
        return {k: s._decimal(v) for k, v in sorted(totals.items())}


def _marker_state_totals(
    prefix: str, households: Sequence[hm.HousingHousehold]
) -> dict[str, int]:
    """Aggregate the raw topcode-marker states, so a total is never read blind.

    A ``TUTILS_*`` marker of 0 is inside its published domain and a ``TRNTMT_*``
    marker of 0 is not. Whether the public file writes 0, a blank or nothing for
    a person outside a topcoded group is undocumented and unread, so the states
    stay separate here and no zero is scored as a cleared non-topcode.
    """
    states = ("zero", "invalid_domain", "blank", "absent")
    return {
        f"{prefix}_topcode_marker_{state}_records": sum(
            int(h.diagnostics[f"{prefix}_topcode_marker_{state}_records"])
            for h in households
        )
        for state in states
    } | {
        f"{prefix}_topcode_marker_{state}_households": sum(
            int(h.diagnostics[f"{prefix}_topcode_marker_{state}_records"]) > 0
            for h in households
        )
        for state in states
    }


def housing_diagnostics(
    households: Sequence[hm.HousingHousehold],
) -> dict:
    """Unweighted support diagnostics, plus explicitly named source-weight totals.

    Every count is unweighted. The weighted block sums the household mean member
    ``WPFINWGT``; it is named for exactly that and is not a publisher household
    sampling weight, so it must not be reported as a household count.
    """
    people = sum(len(h.members) for h in households)
    household_fields = (*hm.HOUSEHOLD_CATEGORIES, *hm.HOUSEHOLD_AMOUNTS)
    admitted = tuple(h for h in households if h.exclusion is None)
    person_month = [v for h in households for v in h.person_month_measurements]
    return {
        "households": len(households),
        "people": people,
        "admitted_unassisted_renter_households": len(admitted),
        "exclusions_exclusive": _counter(
            h.exclusion if h.exclusion is not None else "admitted" for h in households
        ),
        "exclusion_order_exclusive": list(hm.EXCLUSION_ORDER),
        "tenure_classes": _counter(h.tenure for h in households),
        "assistance_classes": _counter(h.assistance for h in households),
        "assistance_universe_conflicts_nonexclusive": sum(
            h.assistance == "assistance_universe_conflict" for h in households
        ),
        # How each household's EVOUCHER evidence was graded against its published
        # universe. Only ``expected_blank_not_in_universe`` is compatible with an
        # explicit ERENTSUB = 2, so this is the audit surface for the admission.
        "voucher_evidence_states": _counter(
            str(h.diagnostics["voucher_evidence"]) for h in households
        ),
        "voucher_evidence_states_admitted": _counter(
            str(h.diagnostics["voucher_evidence"]) for h in admitted
        ),
        "household_codes": {
            field: _code_counter(h.household(field) for h in households)
            for field in hm.HOUSEHOLD_CATEGORIES
        },
        "household_field_classes": {
            field: _counter(
                hm.measurement_class(h.household(field)) for h in households
            )
            for field in household_fields
        },
        "household_field_quality": {
            field: _counter(
                hm.measurement_quality(h.household(field)) for h in households
            )
            for field in household_fields
        },
        "household_copy_conflicts": {
            field: sum(
                "household_copy_conflict" in h.household(field).reasons
                for h in households
            )
            for field in household_fields
        },
        "household_unknown_measurements": {
            field: sum(h.household(field).value is None for h in households)
            for field in household_fields
        },
        "person_month_codes": {
            field: _code_counter(v for v in person_month if v.concept == field)
            for field in hm.PERSON_MONTH_CATEGORIES
        },
        "person_month_quality": {
            field: _counter(
                hm.measurement_quality(v) for v in person_month if v.concept == field
            )
            for field in hm.PERSON_MONTH_CATEGORIES
        },
        "person_month_member_disagreement_households": {
            field: sum(
                bool(h.diagnostics[f"{field.lower()}_member_disagreement"])
                for h in households
            )
            for field in hm.PERSON_MONTH_CATEGORIES
        },
        # The person-month indicators never set a household's class, but a
        # member-reported receipt inside an explicitly unassisted household is a
        # real cross-grain contradiction and is surfaced rather than averaged away.
        "person_month_receipt_in_explicit_unassisted_households": sum(
            h.assistance == "explicit_unassisted"
            and int(h.diagnostics["eehc_rentsub_receipt_members"]) > 0
            for h in households
        ),
        "person_month_receipt_in_admitted_households": sum(
            int(h.diagnostics["eehc_rentsub_receipt_members"]) > 0 for h in admitted
        ),
        "no_person_month_receipt_in_assisted_households": sum(
            h.assistance.startswith("assisted")
            and int(h.diagnostics["eehc_rentsub_receipt_members"]) == 0
            for h in households
        ),
        "rent_quality_admitted": _counter(
            str(h.diagnostics["rent_amount_quality"]) for h in admitted
        ),
        "rent_class_admitted": _counter(
            str(h.diagnostics["rent_amount_class"]) for h in admitted
        ),
        "owner_debt_resolution": _counter(
            str(h.diagnostics["owner_debt"]) for h in households if h.tenure == "owner"
        ),
        # ``owner_debt`` admits an allocated debt indicator, so the contributing
        # evidence classes are reported beside it rather than described as
        # "reported", which in this vocabulary means allocation flag 1.
        "owner_debt_evidence_classes": {
            field: _counter(
                str(h.diagnostics["owner_debt_evidence"][field])
                for h in households
                if h.tenure == "owner"
            )
            for field in ("EPRDEBT", "EMHDEBT")
        },
        "boundaries": {
            "rent_zero_admitted": sum(
                bool(h.diagnostics["rent_zero"]) for h in admitted
            ),
            "rent_at_domain_maximum_admitted": sum(
                bool(h.diagnostics["rent_at_domain_maximum"]) for h in admitted
            ),
            "rent_topcode_marked_households": sum(
                int(h.diagnostics["rent_topcode_marked_records"]) > 0
                for h in households
            ),
            "rent_topcode_partial_marking_households": sum(
                bool(h.diagnostics["rent_topcode_partial_marking"]) for h in households
            ),
            **_marker_state_totals("rent", households),
            "utilities_zero_households": sum(
                bool(h.diagnostics["utilities_zero"]) for h in households
            ),
            "utilities_at_domain_maximum_households": sum(
                bool(h.diagnostics["utilities_at_domain_maximum"]) for h in households
            ),
            "utilities_topcode_marked_households": sum(
                int(h.diagnostics["utilities_topcode_marked_records"]) > 0
                for h in households
            ),
            "utilities_topcode_partial_marking_households": sum(
                bool(h.diagnostics["utilities_topcode_partial_marking"])
                for h in households
            ),
            **_marker_state_totals("utilities", households),
        },
        "person_weights": s._weights(w for h in households for w in h.person_weights),
        "household_weights": s._weights(h.weight for h in households),
        "source_weight_totals": {
            "statistic": "sum_of_household_mean_member_WPFINWGT",
            "basis": "December_2022_mean_all_members",
            "not_a_publisher_household_sampling_weight": True,
            "not_a_rent_model_sampling_weight": True,
            "justification": (
                "Section 6.1 of the requirements asks for weighted shares by "
                "tenure and assistance code; the only weight in this projection "
                "is the published person weight, so its household mean is summed "
                "and named as such rather than presented as a household total."
            ),
            "by_tenure_class": _weighted_by(households, lambda h: h.tenure),
            "by_assistance_class": _weighted_by(households, lambda h: h.assistance),
            "by_exclusion": _weighted_by(
                households, lambda h: h.exclusion or "admitted"
            ),
        },
    }


def _predictor_digest(
    predictors: Sequence[hpred.HouseholdPredictors],
) -> str:
    """Bind the predictor observations separately from the measurement digest.

    A separate key keeps the two surfaces legible: a predictor value that moves
    must move this digest and the receipt, and must not be able to look like a
    change in the measurement evidence the admission ladder read.
    """
    return s._key(
        "sipp-december-housing-predictors/1",
        [
            {
                "key": p.key,
                "members": p.members,
                "household": [s._measurement(v) for v in p.household_measurements],
                "person_age": [s._measurement(v) for v in p.person_age_measurements],
                "maximum_member_age": s._measurement(p.maximum_member_age),
                "status": p.status,
            }
            for p in predictors
        ],
    )


def _measurement_digest(households: Sequence[hm.HousingHousehold]) -> str:
    return s._key(
        "sipp-december-housing-measurements/1",
        [
            {
                "key": h.key,
                "members": h.members,
                "weight": s._measurement(h.weight),
                "person_weights": [s._measurement(w) for w in h.person_weights],
                "household": [s._measurement(v) for v in h.household_measurements],
                "person_month": [
                    s._measurement(v) for v in h.person_month_measurements
                ],
                "person_topcode": [
                    s._measurement(v) for v in h.person_topcode_measurements
                ],
                "rent": s._measurement(h.rent),
                "tenure": h.tenure,
                "assistance": h.assistance,
                "exclusion": h.exclusion,
            }
            for h in households
        ],
    )


def project(
    rows: Sequence[m.SippRow],
    *,
    source: m.SippSource,
    selection: str,
    complete_membership: bool = True,
    target_year: int = 2022,
) -> HousingProjectionResult:
    """Measure December housing from supplied lexical rows under this recipe.

    The caller owns authentication: ``rows`` must already come from the frozen
    authenticated parent, and ``source`` must carry its pins, which are checked
    here. No bytes are opened, nothing is published and no charge is estimated.
    """
    digest = verify_recipe()
    verify_source(source)
    if type(selection) is not str or selection not in s._MODES:
        raise HousingProjectionError("UNSUPPORTED_SELECTION")
    households = hm.december_housing(
        rows,
        source=source,
        complete_membership=complete_membership,
        target_year=target_year,
    )
    # The predictor pass runs strictly after admission and takes the decided
    # households as an input, so nothing it observes can reach ``_exclusion``.
    predictors = hpred.december_predictors(
        rows,
        households,
        source=source,
        complete_membership=complete_membership,
        target_year=target_year,
    )
    evidence = selection_evidence(households, selection)
    diagnostics = dict(housing_diagnostics(households))
    diagnostics["prospective_predictors"] = hpred.predictor_diagnostics(
        predictors, households
    )
    receipt = {
        "format": RECEIPT_FORMAT,
        "recipe_version": RECIPE_VERSION,
        "recipe_sha256": RECIPE_SHA256,
        "recipe_digest": digest,
        "recipe": recipe_document(),
        "implementation": implementation_binding(),
        "source_identity": source_identity(source),
        "selection": selection,
        "selection_evidence": evidence,
        "membership_digest": s._membership(households),
        "measurement_digest": _measurement_digest(households),
        "predictor_digest": _predictor_digest(predictors),
        "diagnostics": diagnostics,
    }
    payload = canonical_json(receipt)
    return HousingProjectionResult(
        households, diagnostics, payload, s._sha(payload), digest, predictors
    )
