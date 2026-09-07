"""Join the annual SIPP panel measure to the frozen December housing projection.

This module is a *seam*, not a producer. It opens no file, authenticates no
byte, publishes nothing, fits nothing, draws nothing, estimates no charge, and
rewrites no recipient income. It takes a panel that
:mod:`microcosm.build.sipp_annual_panel` already read, a
:class:`~microcosm.build.sipp_housing_projection.HousingProjectionResult` that
the frozen recipe already produced, and turns the two into one private,
versioned, closed artifact plus a count-only quality receipt.

Two things it deliberately keeps apart. First, **source authentication**: the
annual panel measures the SHA-256 of the stream it read, while the December
housing projector accepts a caller-asserted frozen-parent identity whose own
receipt says ``supplied_rows_verified_against_these_bytes: False``. Those are
different kinds of fact, so both are recorded, their agreement is a recorded
state rather than a conclusion, and
``authenticated_source_bytes_established_here`` is false whatever they say. A
runner that enumerated the authenticated December roster owns that chain.
Here stream audits themselves are external declarations. Only explicitly
invented evidence may disagree with the frozen source digest.
Second, **selection**: the annual rule is a relative first-32 over the eligible
December identifier index of the streamed source; the housing projector's
``first_32_ssuids`` proves only that the caller supplied no more than 32 SSUIDs.
Both declarations are retained, their selected identities must actually match,
and neither string is treated as the other.

The join is by canonical 14-digit SSUID **and** December residence **and** the
exact ordered member and person-month identity. A shared SSUID alone is not
enough: one sample unit can hold several December residences, and an adult can
have lived somewhere else earlier in the year. December rows the annual panel
could not anchor - an unknown residence, or a row outside monthly membership -
are preserved as unresolved coverage; they are never dropped and never counted
as complete.

See docs/us-sipp-annual-december-housing-evidence.md.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from microcosm.graph.canonical import canonical_json

from . import sipp_annual_panel as ap
from . import sipp_december_source as s
from . import sipp_housing_measurements as hm
from . import sipp_housing_predictors as hpred
from . import sipp_housing_projection as hp
from . import sipp_measurements as m

RECIPE_VERSION = "sipp-annual-december-housing-evidence/1"
ARTIFACT_FORMAT = "sipp_annual_december_housing_evidence_v1"
RECEIPT_FORMAT = ARTIFACT_FORMAT + "_quality_receipt"
QUALITY_SCHEMA = "sipp-annual-december-housing-evidence-quality/1"

# Byte identity of ``recipe_bytes()``. The recipe embeds ``annual_identity()``,
# so an edit to either this contract or the annual source module must move this
# pin in the same commit.
RECIPE_SHA256 = "b73f168b3c3876487fd0c0ae7ddf78c5a71389ea83ed959fe01b5941a5d5a4d3"

# The December housing modules whose bytes decide what the parent receipt means.
# Identical to the set the accepted weight-reference lane pinned at 61f2f499.
FROZEN_HOUSING_MODULE_SHA256 = {
    "microcosm.build.sipp_december_source": "faa97d8699028111de48c7b065e0463def3007e985f1ae5cca35cf13ff129d4c",
    "microcosm.build.sipp_housing_measurements": "0f8bc1171a45c3019487f34d08d05e531b4fdba00856581f4510f9317c278141",
    "microcosm.build.sipp_housing_predictors": "0644ce79e202f05e6e9ef141adf1ec5c132b6c70cbb65fe111cc38de53081ee8",
    "microcosm.build.sipp_housing_projection": "0c5159ef1752a164be20e91e16365fc2c4c4ce981415a2d39cfb58e3e2edad2d",
    "microcosm.build.sipp_measurements": "81fac9866b72cf42dbd98a2eb24a9c1087893ede3fa57c8ccdbd411812510b2c",
}

# The annual source module this seam is bound to. Version 1 (02e7b293) computed
# the annual measure inside ``diagnostics`` and exported no records, so it
# cannot supply this artifact; the pin names the bytes that can.
ANNUAL_MODULE_SHA256 = (
    "784464c12b2ec1a994f43396a4241d9c7bd3fd96a56740cbb3808c0e677f043a"
)

# The ten lexical columns both lanes project. Derived from the two field lists,
# never transcribed, so a column added on either side is compared automatically.
SHARED_COLUMNS = tuple(sorted(set(ap.FIELDS) & set(hp.PROJECTION_FIELDS)))
# The union a future source runner must extract: the annual 67 plus the 28
# December housing additions. Metadata only - nothing here reads, hashes or
# stats a source, and the seven separate weight/reference columns are excluded.
FIELD_UNION = tuple(sorted(set(ap.FIELDS) | set(hp.PROJECTION_FIELDS)))
HOUSING_ADDITIONS = tuple(sorted(set(hp.PROJECTION_FIELDS) - set(ap.FIELDS)))
OPTIONAL_WEIGHT_REFERENCE_ADDITIONS = 7

# Small, explicit ceilings. This seam holds one bounded invented or first-32
# slice in memory; it is not a bulk decoder and never reserves a large buffer.
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_ANNUAL_ROWS = 20_000
MAX_HOUSING_ROWS = 2_000
MAX_HOUSEHOLDS = 512
MAX_UNRESOLVED_ROWS = 2_000
MAX_SELECTED_SSUIDS = 32
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 2_000_000
MAX_JSON_STRING = 4096
MAX_LEXICAL_BYTES = 64

_ARTIFACT_KEYS = frozenset(
    {
        "admission",
        "annual",
        "december_housing",
        "format",
        "implementation",
        "join",
        "quality",
        "recipe",
        "recipe_sha256",
        "selection_binding",
        "source_binding",
    }
)
_ANNUAL_KEYS = frozenset({"identity", "passes", "records", "rows", "selected_ssuids"})
_HOUSING_KEYS = frozenset({"parent", "records", "rows"})
_ANNUAL_ROW_KEYS = frozenset({"key", "values"})
_HOUSING_ROW_KEYS = frozenset({"contradictions", "values"})
_PASS_KEYS = frozenset({"bytes", "rows", "sha256"})
_INDEX_PASS_EXTRA = frozenset(
    {"december_person_rows", "eligible_households", "eligible_ssuids"}
)
_SELECTION_KEYS = frozenset(
    {"annual_limit", "annual_rule", "annual_selected_ssuids", "housing_mode"}
)
_SOURCE_KEYS = frozenset(
    {
        "declared",
        "panel_stream_bytes",
        "panel_stream_rows",
        "panel_stream_sha256",
        "panel_stream_passes",
    }
)


class AnnualHousingSourceError(ValueError):
    """Sanitized refusal at the annual/December housing evidence boundary."""


@dataclass(frozen=True)
class SourceBinding:
    """External source facts. Supplied rows never establish any of them.

    ``declared`` is the frozen-parent identity the December housing recipe
    requires and labels caller-asserted. The ``panel_stream_*`` fields are what
    the annual panel measured while streaming, and they are supplied from
    outside the artifact so that a retained audit block cannot be edited and
    re-digested into agreement with itself.
    """

    declared: m.SippSource
    panel_stream_sha256: str
    panel_stream_bytes: int
    panel_stream_rows: int
    index_december_person_rows: int
    index_eligible_households: int
    index_eligible_ssuids: int
    panel_stream_passes: int = 2
    evidence_origin: str = "supplied_rows_unverified"


@dataclass(frozen=True)
class SelectionBinding:
    """Both selector declarations, kept distinct and both retained."""

    annual_rule: str
    annual_limit: int
    annual_selected_ssuids: tuple[str, ...]
    housing_mode: str


@dataclass(frozen=True)
class AnnualHousingEvidence:
    """The private canonical artifact and its count-only quality receipt."""

    artifact_json: bytes
    artifact_sha256: str
    receipt_json: bytes
    receipt_sha256: str

    @property
    def households(self) -> tuple[dict, ...]:
        return tuple(json.loads(self.artifact_json)["join"]["households"])

    @property
    def quality(self) -> dict:
        return json.loads(self.receipt_json)["quality"]


def recipe_document() -> dict:
    """A detached, source-pinned declaration carrying no self-module hash."""
    return json.loads(
        canonical_json(
            {
                "version": RECIPE_VERSION,
                "artifact_format": ARTIFACT_FORMAT,
                "receipt_format": RECEIPT_FORMAT,
                "annual_identity": ap.annual_identity(),
                "housing_parent": {
                    "recipe_version": hp.RECIPE_VERSION,
                    "recipe_sha256": hp.RECIPE_SHA256,
                    "receipt_format": hp.RECEIPT_FORMAT,
                    "implementation_sha256": FROZEN_HOUSING_MODULE_SHA256,
                    "projection_fields": list(hp.PROJECTION_FIELDS),
                    "admission_ladder": list(hm.EXCLUSION_ORDER),
                    "modules_unchanged_by_this_lane": True,
                },
                "join_contract": {
                    "keys": [
                        "canonical_14_digit_SSUID",
                        "december_ERESIDENCEID",
                        "exact_ordered_member_and_person_month_identity",
                    ],
                    "ssuid_alone_is_insufficient": True,
                    "why": (
                        "one sample unit can contain multiple December "
                        "residences, and adults can have other residences "
                        "earlier in the year"
                    ),
                    "annual_grouping": "live_rows_with_usable_residence",
                    "housing_grouping": "supplied_december_rows",
                    "shared_columns_compared_lexically": list(SHARED_COLUMNS),
                    "unresolved_coverage_states": [
                        "inactive_december_row",
                        "unanchored_live_december_row",
                    ],
                    "unresolved_rows_are_preserved_not_dropped": True,
                    "unresolved_rows_are_never_labelled_complete": True,
                    "mismatched_join_roster": "refused",
                },
                "selection_contract": {
                    "annual_rule": ap.SELECTION,
                    "annual_limit": ap.SELECTION_LIMIT,
                    "annual_rule_declares": (
                        "relative_first_32_of_the_eligible_December_identifier_"
                        "index_over_the_streamed_source"
                    ),
                    "housing_mode_establishes": (
                        "supplied_ssuid_count_no_greater_than_32_caller_asserted"
                    ),
                    "housing_supported_modes": ["first_32_ssuids"],
                    "declarations_are_not_interchangeable": True,
                    "selected_identities_must_match": True,
                    "all_ssuids_is_not_a_256_workaround": True,
                    "a_256_rung_needs_a_new_housing_selection_recipe": True,
                },
                "source_authentication": {
                    "annual_panel_measures": "sha256_of_the_stream_it_read",
                    "housing_projector_asserts": (
                        "frozen_parent_pins_supplied_by_the_caller"
                    ),
                    "declared_identity_must_agree_across_both_lanes": True,
                    "stream_digest_mismatch_allowed_only_for": "invented_only",
                    "relative_selection_authenticated_here": False,
                    "stream_digest_agreement_is_recorded_not_authentication": True,
                    "authenticated_source_bytes_established_here": False,
                    "chain_established_by": (
                        "authenticated_runner_over_the_full_december_roster"
                    ),
                },
                "future_runner_field_union": {
                    "count": len(FIELD_UNION),
                    "fields": list(FIELD_UNION),
                    "annual_field_count": len(ap.FIELDS),
                    "housing_additions": list(HOUSING_ADDITIONS),
                    "verified_by_the_current_67_field_receipt": False,
                    "additional_field_real_data_lexical_behaviour": "unverified",
                    "optional_weight_reference_additions": (
                        OPTIONAL_WEIGHT_REFERENCE_ADDITIONS
                    ),
                    "optional_weight_reference_union_size": (
                        len(FIELD_UNION) + OPTIONAL_WEIGHT_REFERENCE_ADDITIONS
                    ),
                    "optional_weight_reference_additions_excluded_because": (
                        "selecting a fit-weight candidate is outside this slice"
                    ),
                    "no_source_read_hash_or_stat_performed_here": True,
                },
                "arithmetic": ("frozen_measurement_80_digit_half_even_Decimal_context"),
                "complete_member_join": "refuse_any_December_membership_reason",
                "transport_bounds": {
                    "artifact_bytes": MAX_ARTIFACT_BYTES,
                    "annual_rows": MAX_ANNUAL_ROWS,
                    "housing_rows": MAX_HOUSING_ROWS,
                    "json_depth": MAX_JSON_DEPTH,
                    "json_nodes": MAX_JSON_NODES,
                    "json_string": MAX_JSON_STRING,
                    "lexical_bytes": MAX_LEXICAL_BYTES,
                },
                "amount_disclosure": (
                    "amounts_live_only_in_the_private_artifact; "
                    "the quality receipt and every log line are count_only"
                ),
                "non_claims": {
                    "annual_measure_covers_under_15_amounts": False,
                    "annual_measure_equals_ASEC_HTOTVAL": False,
                    "annual_measure_equals_ACS_HINCP": False,
                    "annual_once_per_person_lump_sums_added_to_the_measure": False,
                    "unavailable_annual_measure_reported_as_zero": False,
                    "december_rent_annualized_or_rebased": False,
                    "full_contract_charge_observed_for_assisted_households": False,
                    "supplied_rows_verified_against_authenticated_bytes": False,
                    "housing_selection_recipe_extended_beyond_first_32": False,
                    "weight_reference_fields_included": False,
                    "fit_weight_candidate_selected": False,
                    "recipient_income_rewritten_or_transferred": False,
                    "model_input_or_donor_matrix_bound": False,
                    "target_standard_errors_or_uncertainty_schema_supplied": False,
                },
            }
        )
    )


def recipe_bytes() -> bytes:
    return canonical_json(recipe_document())


def verify_recipe() -> str:
    """Refuse unless this contract, the annual source and the parent all match."""
    hp.verify_recipe()
    if hp.module_digests() != FROZEN_HOUSING_MODULE_SHA256:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_CODE_TAMPERED")
    if ap.annual_identity()["module_sha256"] != ANNUAL_MODULE_SHA256:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ANNUAL_CODE_TAMPERED")
    payload = recipe_bytes()
    if s._sha(payload) != RECIPE_SHA256:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_RECIPE_TAMPERED")
    return s._key("sipp-annual-december-housing-evidence-recipe/1", recipe_document())


def _implementation() -> dict:
    return {
        "housing_parent_modules": dict(FROZEN_HOUSING_MODULE_SHA256),
        "annual_module_sha256": ANNUAL_MODULE_SHA256,
        "annual_contract_sha256": ap.CONTRACT_SHA256,
        "evidence_module_sha256": s._sha(Path(__file__).read_bytes()),
        "canonical_module_sha256": s._sha(
            Path(
                __import__(canonical_json.__module__, fromlist=["__file__"]).__file__
            ).read_bytes()
        ),
    }


def _decimal(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if not value.is_finite():
        raise AnnualHousingSourceError("ANNUAL_HOUSING_NONFINITE_MEASUREMENT")
    return format(value, "f")


def _verify_source_binding(binding: SourceBinding, panel: ap.Panel) -> dict:
    """Bind the panel's measured stream to externally supplied source facts."""
    if not isinstance(binding, SourceBinding):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SOURCE_BINDING_TYPE")
    hp.verify_source(binding.declared)
    if (
        type(binding.declared.survey_year) is not int
        or type(binding.declared.reference_year) is not int
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SOURCE_VINTAGE_TYPE")
    index_counts = {
        "december_person_rows": binding.index_december_person_rows,
        "eligible_households": binding.index_eligible_households,
        "eligible_ssuids": binding.index_eligible_ssuids,
    }
    if (
        type(binding.panel_stream_sha256) is not str
        or not re.fullmatch(r"[0-9a-f]{64}", binding.panel_stream_sha256)
        or type(binding.panel_stream_bytes) is not int
        or type(binding.panel_stream_rows) is not int
        or type(binding.panel_stream_passes) is not int
        or binding.panel_stream_bytes <= 0
        or binding.panel_stream_rows <= 0
        or binding.panel_stream_passes != 2
        or any(type(v) is not int or v < 0 for v in index_counts.values())
        or not (
            len(panel.selected)
            <= binding.index_eligible_ssuids
            <= binding.index_eligible_households
            <= binding.index_december_person_rows
            <= binding.panel_stream_rows
        )
        or binding.panel_stream_rows < len(panel.rows)
        or binding.evidence_origin not in ("invented_only", "supplied_rows_unverified")
        or (
            binding.evidence_origin != "invented_only"
            and binding.panel_stream_sha256 != binding.declared.data_sha256
        )
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SOURCE_BINDING")
    audits = (panel.index_audit, panel.panel_audit)
    for audit in audits:
        if (
            not isinstance(audit, Mapping)
            or audit.get("sha256") != binding.panel_stream_sha256
            or audit.get("bytes") != binding.panel_stream_bytes
            or audit.get("rows") != binding.panel_stream_rows
        ):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_PANEL_STREAM_DISAGREEMENT")
    retained_december = [r for k, r in panel.rows.items() if k[2] == 12]
    retained_eligible_households = {
        (r["SSUID"], int(ap.usable(r, "ERESIDENCEID")[0]))
        for r in retained_december
        if ap.eligible_id(r)
    }
    if binding.index_december_person_rows < len(
        retained_december
    ) or binding.index_eligible_households < len(retained_eligible_households):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PANEL_INDEX_COVERAGE")
    if any(panel.index_audit.get(k) != v for k, v in index_counts.items()):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PANEL_INDEX_DISAGREEMENT")
    return {
        "declared": hp.source_identity(binding.declared),
        "panel_stream_sha256": binding.panel_stream_sha256,
        "panel_stream_bytes": binding.panel_stream_bytes,
        "panel_stream_rows": binding.panel_stream_rows,
        "panel_stream_passes": binding.panel_stream_passes,
        "index_counts": index_counts,
        "evidence_origin": binding.evidence_origin,
        "declared_source_matches_panel_stream_digest": (
            binding.panel_stream_sha256 == binding.declared.data_sha256
        ),
        "stream_digest_agreement_is_not_authentication": True,
        "authenticated_source_bytes_established_here": False,
    }


def _verify_selection(
    binding: SelectionBinding,
    panel: ap.Panel,
    parent: hp.HousingProjectionResult,
    declared_mode: str,
) -> dict:
    """Keep the annual rule and the housing mode apart, and match their identities."""
    if not isinstance(binding, SelectionBinding):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_BINDING_TYPE")
    if binding.annual_rule != ap.SELECTION:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_RULE")
    if (
        type(binding.annual_limit) is not int
        or binding.annual_limit != ap.SELECTION_LIMIT
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_LIMIT")
    if (
        binding.housing_mode != "first_32_ssuids"
        or binding.housing_mode != declared_mode
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_MODE")
    if type(binding.annual_selected_ssuids) is not tuple or any(
        type(v) is not str for v in binding.annual_selected_ssuids
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_IDENTITY")
    selected = binding.annual_selected_ssuids
    if (
        selected != panel.selected
        or len(selected) > MAX_SELECTED_SSUIDS
        or selected != tuple(sorted(set(selected)))
        or any(not re.fullmatch(r"[0-9]{14}", value) for value in selected)
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_IDENTITY")
    if len(selected) != min(ap.SELECTION_LIMIT, panel.index_audit["eligible_ssuids"]):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_INDEX_COUNT")
    eligible = {
        identity[0]
        for identity, row in panel.rows.items()
        if identity[2] == 12 and ap.eligible_id(row)
    }
    if set(selected) != eligible:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_ELIGIBILITY")
    if set(selected) != {k[0] for k in panel.rows}:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_IDENTITY")
    if set(selected) != {h.key[0] for h in parent.households}:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_IDENTITY")
    return {
        "annual": {
            "rule": binding.annual_rule,
            "limit": binding.annual_limit,
            "selected_ssuids": list(selected),
            "selected_ssuid_count": len(selected),
            "eligible_ssuids_seen_by_the_index_pass": panel.index_audit.get(
                "eligible_ssuids"
            ),
            "amounts_used_for_selection": False,
            "declares": ("relative_first_32_of_the_eligible_December_identifier_index"),
            "relative_selection_authenticated_here": False,
            "source_wide_selection_evidence": "externally_supplied_unverified",
        },
        "housing": {
            "mode": binding.housing_mode,
            "establishes": "supplied_ssuid_count_no_greater_than_32_caller_asserted",
        },
        "declarations_are_not_interchangeable": True,
        "selected_identities_matched": True,
    }


def _verify_housing_parent(
    parent: hp.HousingProjectionResult, expected: str, source: m.SippSource
) -> tuple[dict, str]:
    """Recompute the whole parent receipt; a named SHA alone is not enough."""
    hp.verify_source(source)
    if not isinstance(parent, hp.HousingProjectionResult):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_TYPE")
    if type(expected) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_RECEIPT_SHA")
    if (
        type(parent.receipt_json) is not bytes
        or len(parent.receipt_json) > MAX_ARTIFACT_BYTES
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_JSON")
    if s._sha(parent.receipt_json) != expected or parent.receipt_sha256 != expected:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_RECEIPT_SHA")
    document = _bounded_json(parent.receipt_json)
    if type(document) is not dict:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_JSON")
    diagnostics = dict(hp.housing_diagnostics(parent.households))
    diagnostics["prospective_predictors"] = hpred.predictor_diagnostics(
        parent.predictors, parent.households
    )
    mode = document.get("selection")
    check = {
        "format": hp.RECEIPT_FORMAT,
        "recipe_version": hp.RECIPE_VERSION,
        "recipe_sha256": hp.RECIPE_SHA256,
        "recipe_digest": hp.recipe_digest(),
        "recipe": hp.recipe_document(),
        "implementation": hp.implementation_binding(),
        "source_identity": hp.source_identity(source),
        "selection": mode,
        "selection_evidence": hp.selection_evidence(parent.households, mode),
        "membership_digest": s._membership(parent.households),
        "measurement_digest": hp._measurement_digest(parent.households),
        "predictor_digest": hp._predictor_digest(parent.predictors),
        "diagnostics": diagnostics,
    }
    if (
        canonical_json(check) != parent.receipt_json
        or canonical_json(parent.diagnostics) != canonical_json(diagnostics)
        or parent.recipe_digest != hp.recipe_digest()
        or mode not in s._MODES
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_CONTENT")
    return {
        "receipt_sha256": expected,
        "receipt_format": hp.RECEIPT_FORMAT,
        **{
            key: document[key]
            for key in (
                "recipe_version",
                "recipe_sha256",
                "membership_digest",
                "measurement_digest",
                "predictor_digest",
                "source_identity",
                "selection",
                "selection_evidence",
            )
        },
        "parent_modules_unchanged": True,
    }, mode


def _annual_record_json(record: ap.AnnualDecemberHousehold) -> dict:
    """Serialize one annual record, amounts included; this artifact stays private."""
    return {
        "key": [record.key[0], record.key[1]],
        "members": [list(person) for person in record.members],
        "december_person_months": [list(k) for k in record.december_person_months],
        "contributors": [list(person) for person in record.contributors],
        "children_under15": [list(person) for person in record.children_under15],
        "unknown_age_members": [list(p) for p in record.unknown_age_members],
        "contributor_months": [
            {
                "person": list(month.person),
                "month": month.month,
                "household_status": month.household_status,
                "frame": month.frame,
                "residence": month.residence,
                "residence_state": month.residence_state,
                "age": month.age,
                "scope": month.scope,
                "income_status_code": month.income_status_code,
                "income_state": month.income_state,
                "amount": _decimal(month.amount),
            }
            for month in record.contributor_months
        ],
        "contributor_months_present": [
            {"person": list(person), "months": list(months)}
            for person, months in sorted(record.contributor_months_present.items())
        ],
        "membership_reasons": list(record.membership_reasons),
        "reasons": list(record.reasons),
        "amount": _decimal(record.amount),
        "amount_available": record.amount is not None,
        "weight": s._measurement(record.weight),
        "weight_state": record.weight_state,
        "scope": (
            "December_age15plus_persons_over_all_their_months_and_residences; "
            "under15_amounts_not_added"
        ),
    }


def _housing_record_json(
    household: hm.HousingHousehold, predictors: hpred.HouseholdPredictors
) -> dict:
    return {
        "key": [household.key[0], household.key[1]],
        "members": [list(person) for person in household.members],
        "weight": s._measurement(household.weight),
        "person_weights": [s._measurement(w) for w in household.person_weights],
        "household": [s._measurement(v) for v in household.household_measurements],
        "person_month": [
            s._measurement(v) for v in household.person_month_measurements
        ],
        "person_topcode": [
            s._measurement(v) for v in household.person_topcode_measurements
        ],
        "rent": s._measurement(household.rent),
        "utilities": s._measurement(household.utilities),
        "tenure": household.tenure,
        "assistance": household.assistance,
        "exclusion": household.exclusion,
        "predictors": {
            "household": [s._measurement(v) for v in predictors.household_measurements],
            "person_age": [
                s._measurement(v) for v in predictors.person_age_measurements
            ],
            "maximum_member_age": s._measurement(predictors.maximum_member_age),
            "maximum_member_age_lower_bound": s._measurement(
                predictors.maximum_member_age_lower_bound
            ),
            "status": predictors.status,
            "lower_bound_is_not_the_known_maximum": True,
        },
        "amount_period": "December_2022_monthly",
        "annualization": "not_applied",
    }


def _december_coverage(
    panel: ap.Panel, state: ap.PanelState
) -> tuple[dict[tuple[str, int, int], tuple[str, int]], list[dict]]:
    """Split every December person-month into anchored evidence or unresolved."""
    anchored: dict[tuple[str, int, int], tuple[str, int]] = {}
    for household, rows in state.groups.items():
        if household[2] != 12:
            continue
        for row in rows:
            anchored[ap.key(row)] = (household[0], household[1])
    unresolved = []
    for identity, row in sorted(panel.rows.items()):
        if identity[2] != 12 or identity in anchored:
            continue
        residence, residence_state = ap.usable(row, "ERESIDENCEID")
        status = ap.number(row["THHLDSTATUS"], "THHLDSTATUS")
        unresolved.append(
            {
                "person_month": list(identity),
                "state": (
                    "unanchored_live_december_row"
                    if ap.live(row)
                    else "inactive_december_row"
                ),
                "household_status": None if status is None else int(status),
                "residence_state": residence_state,
                "residence": None if residence is None else int(residence),
                "counted_as_complete": False,
                "counted_in_any_joined_household": False,
            }
        )
        if len(unresolved) > MAX_UNRESOLVED_ROWS:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_UNRESOLVED_CEILING")
    return anchored, unresolved


def _join(
    panel: ap.Panel,
    state: ap.PanelState,
    records: Sequence[ap.AnnualDecemberHousehold],
    parent: hp.HousingProjectionResult,
    housing_rows: Sequence[m.SippRow],
) -> tuple[list[dict], list[dict]]:
    """Match the two lanes on SSUID, December residence and person-month identity."""
    anchored, unresolved = _december_coverage(panel, state)
    supplied: dict[tuple[str, int, int], m.SippRow] = {}
    for row in housing_rows:
        values = row.values
        if not isinstance(values, Mapping):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_ROW_TYPE")
        if any(type(value) is not str for value in values.values()):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_LEXICAL_TYPE")
        if set(values) != set(hp.PROJECTION_FIELDS):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_ROW_FIELDS")
        ssuid = values["SSUID"]
        if not re.fullmatch(r"[0-9]{14}", ssuid):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_NONCANONICAL_SSUID")
        _, person = m._number(values, "PNUM", 101, 499, -999)
        _, month = m._number(values, "MONTHCODE", 1, 12, -9)
        if person is None or month is None:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_PERSON_MONTH_KEY")
        if int(month) != 12:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_ROW_OUTSIDE_DECEMBER")
        identity = (ssuid, int(person), 12)
        if identity in supplied:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_DUPLICATE_PERSON_MONTH")
        supplied[identity] = row
    if set(supplied) != set(anchored):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_JOIN_ROSTER_DISAGREEMENT")
    housing_by_key = {h.key: h for h in parent.households}
    predictors_by_key = {p.key: p for p in parent.predictors}
    if len(housing_by_key) != len(parent.households) or set(housing_by_key) != set(
        predictors_by_key
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_ROSTER_DISAGREEMENT")
    if {r.key for r in records} != set(housing_by_key):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_JOIN_ROSTER_DISAGREEMENT")
    households = []
    for record in records:
        if record.membership_reasons:
            raise AnnualHousingSourceError(
                "ANNUAL_HOUSING_INCOMPLETE_DECEMBER_MEMBERSHIP"
            )
        household = housing_by_key[record.key]
        if household.members != record.members:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_MEMBER_ORDER_DISAGREEMENT")
        for identity in record.december_person_months:
            if anchored[identity] != record.key:
                raise AnnualHousingSourceError("ANNUAL_HOUSING_RESIDENCE_DISAGREEMENT")
            annual_row = panel.rows[identity]
            housing_row = supplied[identity].values
            for column in SHARED_COLUMNS:
                if housing_row[column] != annual_row[column]:
                    raise AnnualHousingSourceError(
                        "ANNUAL_HOUSING_SHARED_COLUMN_DISAGREEMENT"
                    )
        households.append(
            {
                "key": [record.key[0], record.key[1]],
                "members": [list(person) for person in record.members],
                "december_person_months": [
                    list(k) for k in record.december_person_months
                ],
                "annual_amount_available": record.amount is not None,
                "annual_reasons": list(record.reasons),
                "annual_weight_state": record.weight_state,
                "annual_contributor_count": len(record.contributors),
                "annual_under15_member_count": len(record.children_under15),
                "housing_tenure": household.tenure,
                "housing_assistance": household.assistance,
                "housing_exclusion": household.exclusion,
                "predictor_status": predictors_by_key[record.key].status,
                "shared_columns_agree": True,
                "joined_on": [
                    "canonical_14_digit_SSUID",
                    "december_ERESIDENCEID",
                    "exact_ordered_member_and_person_month_identity",
                ],
            }
        )
        if len(households) > MAX_HOUSEHOLDS:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_HOUSEHOLD_CEILING")
    return households, unresolved


def _quality(
    records: Sequence[ap.AnnualDecemberHousehold],
    parent: hp.HousingProjectionResult,
    households: Sequence[dict],
    unresolved: Sequence[dict],
    selection: Mapping[str, object],
    source: Mapping[str, object],
) -> dict:
    """Counts only. No amount, and no identifier, reaches this receipt."""
    causes: Counter[str] = Counter()
    signs: Counter[str] = Counter({"negative": 0, "positive": 0, "zero": 0})
    scopes: Counter[str] = Counter()
    states: Counter[str] = Counter()
    weights: Counter[str] = Counter({"positive": 0, "unknown": 0, "zero": 0})
    complete = known_weight = contributors = children = unknown_age = 0
    for record in records:
        weights[record.weight_state] += 1
        contributors += len(record.contributors)
        children += len(record.children_under15)
        unknown_age += len(record.unknown_age_members)
        for month in record.contributor_months:
            scopes[month.scope] += 1
            states[month.income_state] += 1
        if record.reasons:
            causes.update(record.reasons)
            continue
        complete += 1
        signs[
            "negative"
            if record.amount < 0
            else "zero"
            if record.amount == 0
            else "positive"
        ] += 1
        if record.weight.value is not None:
            known_weight += 1
    parent_diagnostics = parent.diagnostics
    per_ssuid = Counter(key[0] for key, _ in ((h.key, h) for h in parent.households))
    return {
        "schema": QUALITY_SCHEMA,
        "annual": {
            "december_households": len(records),
            "complete_households": complete,
            "incomplete_households": len(records) - complete,
            "complete_households_with_known_weight": known_weight,
            "positive_weight_households": weights["positive"],
            "zero_weight_households": weights["zero"],
            "unknown_weight_households": weights["unknown"],
            "contributors": contributors,
            "under15_members_retained": children,
            "unknown_age_members": unknown_age,
            "incomplete_causes": dict(sorted(causes.items())),
            "complete_household_sign_classes": dict(sorted(signs.items())),
            "contributor_month_scopes": dict(sorted(scopes.items())),
            "contributor_month_income_states": dict(sorted(states.items())),
            "amounts_reported_here": False,
        },
        "december_housing": {
            "households": parent_diagnostics["households"],
            "people": parent_diagnostics["people"],
            "admitted_unassisted_renter_households": parent_diagnostics[
                "admitted_unassisted_renter_households"
            ],
            "exclusions_exclusive": parent_diagnostics["exclusions_exclusive"],
            "tenure_classes": parent_diagnostics["tenure_classes"],
            "assistance_classes": parent_diagnostics["assistance_classes"],
            "voucher_evidence_states": parent_diagnostics["voucher_evidence_states"],
            "predictor_status": parent_diagnostics["prospective_predictors"][
                "predictor_status_exclusive"
            ],
            "receipt_unchanged_by_this_lane": True,
        },
        "join": {
            "joined_households": len(households),
            "joined_december_person_months": sum(
                len(h["december_person_months"]) for h in households
            ),
            "annual_only_households": 0,
            "housing_only_households": 0,
            "unresolved_december_rows": dict(
                sorted(Counter(row["state"] for row in unresolved).items())
            ),
            "unresolved_december_row_count": len(unresolved),
            "shared_columns_compared": len(SHARED_COLUMNS),
            "multi_residence_ssuids": sum(n > 1 for n in per_ssuid.values()),
            "selected_ssuid_count": selection["annual"]["selected_ssuid_count"],
        },
        "source": {
            "declared_source_matches_panel_stream_digest": source[
                "declared_source_matches_panel_stream_digest"
            ],
            "authenticated_source_bytes_established_here": False,
            "panel_source_passes": source["panel_stream_passes"],
        },
    }


def _admission() -> dict:
    return {
        "fit_admission": None,
        "release_admission": None,
        "fit_weight_selected": False,
        "model_input_bound": False,
        "recipient_transfer_performed": False,
        "charge_estimated": False,
        "draw_performed": False,
        "annualization_applied": False,
        "price_rebasing_applied": False,
        "genuine_source_bytes_authenticated_here": False,
        "public_release_of_amounts": False,
        "launch_readiness_declared": False,
    }


def project(
    panel: ap.Panel,
    *,
    housing: hp.HousingProjectionResult,
    expected_housing_receipt_sha256: str,
    source: SourceBinding,
    selection_binding: SelectionBinding,
    housing_rows: Sequence[m.SippRow],
) -> AnnualHousingEvidence:
    """Bind one annual panel to one unchanged December housing projection.

    ``panel`` and ``housing_rows`` are supplied evidence; ``housing`` must be the
    projection those same rows produced under the frozen recipe, and its receipt
    is recomputed here rather than trusted. Nothing is opened, nothing is
    published and no admission is granted.
    """
    verify_recipe()
    if not isinstance(panel, ap.Panel):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PANEL_TYPE")
    if type(panel.rows) is not dict or type(panel.selected) is not tuple:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PANEL_TYPE")
    if any(
        type(k) is not tuple
        or len(k) != 3
        or type(k[0]) is not str
        or any(type(v) is not int for v in k[1:])
        for k in panel.rows
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ANNUAL_ROW_KEY")
    if type(housing_rows) not in (list, tuple) or any(
        not isinstance(r, m.SippRow) or type(r.contradictions) is not tuple
        for r in housing_rows
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ROW_TYPE")
    if len(panel.rows) > MAX_ANNUAL_ROWS:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ANNUAL_ROW_CEILING")
    if len(housing_rows) > MAX_HOUSING_ROWS:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_HOUSING_ROW_CEILING")
    # Creation and replay use the same closed lexical grammar. A mutable
    # caller Panel is evidence, not an authenticated source object.
    panel = _rebuild_panel(
        {
            "identity": ap.annual_identity(),
            "selected_ssuids": list(panel.selected),
            "passes": [panel.index_audit, panel.panel_audit],
            "rows": [{"key": list(k), "values": v} for k, v in panel.rows.items()],
            "records": [],
        }
    )
    housing_rows = _rebuild_housing_rows(
        {
            "parent": {},
            "records": [],
            "rows": [
                {"values": r.values, "contradictions": list(r.contradictions)}
                for r in housing_rows
            ],
        }
    )
    source_block = _verify_source_binding(source, panel)
    parent, mode = _verify_housing_parent(
        housing, expected_housing_receipt_sha256, source.declared
    )
    recomputed = hp.project(housing_rows, source=source.declared, selection=mode)
    if recomputed != housing:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_RECOMPUTATION")
    selection = _verify_selection(selection_binding, panel, housing, mode)
    state = ap.panel_state(panel)
    records = ap.annual_december_households(
        panel, source.declared.data_sha256, state=state
    )
    households, unresolved = _join(panel, state, records, housing, housing_rows)
    quality = _quality(
        records, housing, households, unresolved, selection, source_block
    )
    implementation = _implementation()
    artifact = canonical_json(
        {
            "format": ARTIFACT_FORMAT,
            "recipe_sha256": RECIPE_SHA256,
            "recipe": recipe_document(),
            "implementation": implementation,
            "source_binding": source_block,
            "selection_binding": selection,
            "annual": {
                "identity": ap.annual_identity(),
                "selected_ssuids": list(panel.selected),
                "passes": [dict(panel.index_audit), dict(panel.panel_audit)],
                "rows": [
                    {"key": list(identity), "values": dict(row)}
                    for identity, row in sorted(panel.rows.items())
                ],
                "records": [_annual_record_json(r) for r in records],
            },
            "december_housing": {
                "parent": parent,
                "rows": [
                    {
                        "values": dict(row.values),
                        "contradictions": list(row.contradictions),
                    }
                    for row in housing_rows
                ],
                "records": [
                    _housing_record_json(h, p)
                    for h, p in zip(housing.households, housing.predictors, strict=True)
                ],
            },
            "join": {"households": households, "unresolved_coverage": unresolved},
            "quality": quality,
            "admission": _admission(),
        }
    )
    if len(artifact) > MAX_ARTIFACT_BYTES:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_BYTES")
    _bounded_document(artifact, s._sha(artifact))
    receipt = canonical_json(
        {
            "format": RECEIPT_FORMAT,
            "recipe_version": RECIPE_VERSION,
            "recipe_sha256": RECIPE_SHA256,
            "artifact_sha256": s._sha(artifact),
            "artifact_bytes": len(artifact),
            "implementation": implementation,
            "annual_identity": ap.annual_identity(),
            "housing_parent": parent,
            "source_binding": {
                key: value
                for key, value in source_block.items()
                if key != "panel_stream_sha256"
            },
            "selection": {
                "annual_rule": selection["annual"]["rule"],
                "annual_limit": selection["annual"]["limit"],
                "annual_selected_ssuid_count": selection["annual"][
                    "selected_ssuid_count"
                ],
                "annual_selected_ssuid_digest": s._key(
                    "sipp-annual-december-selected-ssuids/1", list(panel.selected)
                ),
                "housing_mode": selection["housing"]["mode"],
                "declarations_are_not_interchangeable": True,
            },
            "quality": quality,
            "admission": _admission(),
            "count_only": True,
            "amounts_present": False,
        }
    )
    return AnnualHousingEvidence(artifact, s._sha(artifact), receipt, s._sha(receipt))


def _no_duplicates(pairs):
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_DUPLICATE_KEY")
    return dict(pairs)


def _refuse_constant(name):
    raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_NONFINITE")


def _exact_keys(value: object, keys: frozenset[str], code: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise AnnualHousingSourceError(code)
    return value


def _bounded_json(payload: bytes) -> dict:
    # Scan nesting before the recursive stdlib parser. Braces inside JSON
    # strings do not contribute to depth. Total bytes cap parser allocation.
    depth = 0
    quoted = escaped = False
    for byte in payload:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise AnnualHousingSourceError("ANNUAL_HOUSING_JSON_DEPTH")
        elif byte in (93, 125):
            depth -= 1
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_refuse_constant,
        )
        stack = [document]
        count = 0
        while stack:
            value = stack.pop()
            count += 1
            if count > MAX_JSON_NODES:
                raise AnnualHousingSourceError("ANNUAL_HOUSING_JSON_NODES")
            if type(value) is str:
                if len(value.encode("utf-8")) > MAX_JSON_STRING:
                    raise AnnualHousingSourceError("ANNUAL_HOUSING_JSON_STRING")
            elif type(value) is float and not math.isfinite(value):
                raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_NONFINITE")
            elif type(value) is dict:
                stack.extend(value.keys())
                stack.extend(value.values())
            elif type(value) is list:
                stack.extend(value)
        if canonical_json(document) != payload:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_NONCANONICAL")
        return document
    except AnnualHousingSourceError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeError) as from_exception:
        raise AnnualHousingSourceError(
            "ANNUAL_HOUSING_ARTIFACT_FORMAT"
        ) from from_exception


def _bounded_document(payload: bytes, expected_sha256: str) -> dict:
    if type(payload) is not bytes:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_TYPE")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_BYTES")
    if type(expected_sha256) is not str or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_SHA")
    if s._sha(payload) != expected_sha256:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_SHA")
    document = _bounded_json(payload)
    _exact_keys(document, _ARTIFACT_KEYS, "ANNUAL_HOUSING_ARTIFACT_FORMAT")
    if document["format"] != ARTIFACT_FORMAT:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_FORMAT")
    return document


def _lexical_values(values: object, fields: tuple[str, ...], code: str) -> dict:
    if type(values) is not dict or set(values) != set(fields):
        raise AnnualHousingSourceError(code)
    for value in values.values():
        if type(value) is not str:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_LEXICAL_TYPE")
        try:
            if len(value.encode("ascii")) > MAX_LEXICAL_BYTES:
                raise AnnualHousingSourceError("ANNUAL_HOUSING_LEXICAL_LENGTH")
        except UnicodeError as exc:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_LEXICAL_ENCODING") from exc
    return dict(values)


def _rebuild_panel(block: dict) -> ap.Panel:
    _exact_keys(block, _ANNUAL_KEYS, "ANNUAL_HOUSING_ARTIFACT_FORMAT")
    rows_in = block["rows"]
    if type(rows_in) is not list or len(rows_in) > MAX_ANNUAL_ROWS:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ANNUAL_ROW_CEILING")
    rows: dict[tuple[str, int, int], dict] = {}
    for entry in rows_in:
        _exact_keys(entry, _ANNUAL_ROW_KEYS, "ANNUAL_HOUSING_ARTIFACT_FORMAT")
        values = _lexical_values(
            entry["values"], ap.FIELDS, "ANNUAL_HOUSING_ANNUAL_ROW_FIELDS"
        )
        identity = ap.key(values)
        stored_key = entry["key"]
        if (
            type(stored_key) is not list
            or len(stored_key) != 3
            or type(stored_key[0]) is not str
            or any(type(v) is not int for v in stored_key[1:])
            or stored_key != list(identity)
        ):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_ANNUAL_ROW_KEY")
        if identity in rows:
            raise AnnualHousingSourceError("ANNUAL_HOUSING_DUPLICATE_PERSON_MONTH")
        rows[identity] = values
    passes = block["passes"]
    if type(passes) is not list or len(passes) != 2:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_FORMAT")
    _exact_keys(
        passes[0], _PASS_KEYS | _INDEX_PASS_EXTRA, "ANNUAL_HOUSING_ARTIFACT_FORMAT"
    )
    _exact_keys(passes[1], _PASS_KEYS, "ANNUAL_HOUSING_ARTIFACT_FORMAT")
    for audit in passes:
        if any(type(v) is not int or v < 0 for k, v in audit.items() if k != "sha256"):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_PANEL_AUDIT_TYPE")
    selected = block["selected_ssuids"]
    if type(selected) is not list or any(type(v) is not str for v in selected):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_IDENTITY")
    return ap.Panel(rows, tuple(selected), passes[0], passes[1])


def _rebuild_housing_rows(block: dict) -> list[m.SippRow]:
    _exact_keys(block, _HOUSING_KEYS, "ANNUAL_HOUSING_ARTIFACT_FORMAT")
    rows_in = block["rows"]
    if type(rows_in) is not list or len(rows_in) > MAX_HOUSING_ROWS:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_HOUSING_ROW_CEILING")
    rows = []
    for entry in rows_in:
        _exact_keys(entry, _HOUSING_ROW_KEYS, "ANNUAL_HOUSING_ARTIFACT_FORMAT")
        values, contradictions = entry["values"], entry["contradictions"]
        values = _lexical_values(
            values, hp.PROJECTION_FIELDS, "ANNUAL_HOUSING_ROW_FIELDS"
        )
        # Preserve bounded caller contradiction codes privately; they carry no
        # source-authentication authority and never enter the public receipt.
        if (
            type(contradictions) is not list
            or len(contradictions) > len(hp.PROJECTION_FIELDS)
            or any(
                type(v) is not str or not re.fullmatch(r"[a-zA-Z0-9_:]{1,128}", v)
                for v in contradictions
            )
            or len(contradictions) != len(set(contradictions))
        ):
            raise AnnualHousingSourceError("ANNUAL_HOUSING_CONTRADICTIONS")
        rows.append(m.SippRow(values, tuple(contradictions)))
    return rows


def _rebuild_selection(block: dict, mode: str) -> SelectionBinding:
    if not isinstance(block, dict) or set(block) != {
        "annual",
        "housing",
        "declarations_are_not_interchangeable",
        "selected_identities_matched",
    }:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_FORMAT")
    annual = block["annual"]
    if (
        not isinstance(annual, dict)
        or type(annual.get("selected_ssuids")) is not list
        or any(type(v) is not str for v in annual["selected_ssuids"])
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_FORMAT")
    return SelectionBinding(
        annual.get("rule"),
        annual.get("limit"),
        tuple(annual.get("selected_ssuids") or ()),
        mode,
    )


def decode_artifact(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_source_binding: SourceBinding,
    expected_recipe_sha256: str,
    expected_selection_binding: SelectionBinding,
    expected_housing_receipt_sha256: str,
) -> AnnualHousingEvidence:
    """Check external expectations, then recompute all encoded derived values.

    Every derived record - the annual measure, the December measurements, the
    predictors, the join and the quality counts - is recomputed from the
    retained lexical rows and the recomputed parent projection, and the result
    must reproduce the supplied bytes exactly. Retained facts that cannot be
    recomputed, namely the two stream audits and the declared source identity,
    are anchored instead to ``expected_source_binding``, which comes from
    outside the artifact.
    """
    if expected_recipe_sha256 != RECIPE_SHA256:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_RECIPE_EXPECTATION")
    document = _bounded_document(payload, expected_sha256)
    if document["recipe_sha256"] != expected_recipe_sha256:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_RECIPE_EXPECTATION")
    _verify_source_binding_expectation(
        document["source_binding"], expected_source_binding
    )
    panel = _rebuild_panel(document["annual"])
    housing_rows = _rebuild_housing_rows(document["december_housing"])
    parent_block = document["december_housing"]["parent"]
    if not isinstance(parent_block, dict):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_FORMAT")
    mode = parent_block.get("selection")
    expected_parent = expected_housing_receipt_sha256
    if parent_block.get("receipt_sha256") != expected_parent:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_RECEIPT_SHA")
    selection = _rebuild_selection(document["selection_binding"], mode)
    if selection != expected_selection_binding:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_EXPECTATION")
    if mode != "first_32_ssuids":
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SELECTION_MODE")
    try:
        parent = hp.project(
            housing_rows, source=expected_source_binding.declared, selection=mode
        )
    except (hp.HousingProjectionError, m.MeasurementError) as exc:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_PARENT_RECOMPUTATION") from exc
    result = project(
        panel,
        housing=parent,
        expected_housing_receipt_sha256=expected_parent,
        source=expected_source_binding,
        selection_binding=expected_selection_binding,
        housing_rows=housing_rows,
    )
    if result.artifact_json != payload:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_RECOMPUTATION")
    return result


def _verify_source_binding_expectation(
    recorded: object, expected: SourceBinding
) -> None:
    if not isinstance(expected, SourceBinding):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SOURCE_BINDING_TYPE")
    hp.verify_source(expected.declared)
    if not isinstance(recorded, dict) or set(recorded) != {
        "declared",
        "panel_stream_sha256",
        "panel_stream_bytes",
        "panel_stream_rows",
        "panel_stream_passes",
        "index_counts",
        "evidence_origin",
        "declared_source_matches_panel_stream_digest",
        "stream_digest_agreement_is_not_authentication",
        "authenticated_source_bytes_established_here",
    }:
        raise AnnualHousingSourceError("ANNUAL_HOUSING_ARTIFACT_FORMAT")
    if (
        recorded["declared"] != hp.source_identity(expected.declared)
        or recorded["panel_stream_sha256"] != expected.panel_stream_sha256
        or recorded["panel_stream_bytes"] != expected.panel_stream_bytes
        or recorded["panel_stream_rows"] != expected.panel_stream_rows
        or recorded["panel_stream_passes"] != expected.panel_stream_passes
        or recorded["evidence_origin"] != expected.evidence_origin
        or recorded["index_counts"]
        != {
            "december_person_rows": expected.index_december_person_rows,
            "eligible_households": expected.index_eligible_households,
            "eligible_ssuids": expected.index_eligible_ssuids,
        }
    ):
        raise AnnualHousingSourceError("ANNUAL_HOUSING_SOURCE_BINDING_EXPECTATION")
