"""Canonical producer-input DAG for the US stacked late stage.

This module is the data-only production declaration that binds the primary
PUF pass, the sixteen post-clone source operators, and the nineteen bounded
late-transfer groups.  It deliberately separates two kinds of input data:

* :class:`ProducerInput` values are the role-scoped dependencies which can
  affect late-stage scheduling; and
* :data:`US_LATE_SOURCE_INPUT_INVENTORIES` records each source kernel's full
  effective raw, structural, weight, and optional-input surface.

Keeping the full inventories beside the executable graph makes an empty
late-dependency set explicit without pretending that a source kernel has no
inputs.  The graph itself is validated and topologically sorted at import.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from microcosm.build.us_runtime.acs_income_universe import (
    ACS_PUMS_EARNINGS_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.acs_transfer import TargetFamilies
from microcosm.build.us_runtime.education_inputs import (
    US_EDUCATION_INPUTS_OUTPUT_COLUMNS,
    US_EDUCATION_INPUTS_OWNED_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerInputColumn,
    ProducerOutput,
    ProducerSchedule,
    derive_producer_schedule,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_DEFERRED_TRANSFER_INPUTS,
    POOL_OPERATOR_CONTRACTS,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    pool_post_puf_puf_producer_target_families,
    pool_post_puf_transfer_target_families,
)
from microcosm.build.us_runtime.operator_boundary import (
    FORMULA_OWNED_SOURCE_COLUMNS,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    US_LATE_OVERLAP_OWNERSHIP_TARGETS,
    US_LATE_SOURCE_CALLBACK_PASSTHROUGH_OUTPUTS,
    us_late_overlap_ownership_receipt,
)

__all__ = [
    "CANONICAL_US_LATE_PRODUCER_REGISTRY",
    "CANONICAL_US_LATE_PRODUCER_SCHEDULE",
    "CANONICAL_US_LATE_SOURCE_OUTPUTS",
    "CANONICAL_US_LATE_TRANSFER_GROUPS",
    "EffectiveInputRequirement",
    "ScopedInput",
    "SourceInputInventory",
    "TransferProducerGroup",
    "US_LATE_EXTERNAL_STAGES",
    "US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT",
    "US_LATE_ACS_EARNINGS_UNIVERSE_INPUT_INVENTORY",
    "US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT",
    "US_LATE_ACS_EARNINGS_UNIVERSE_STAGE",
    "US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT",
    "US_LATE_PRIMARY_PUF_STAGE",
    "US_LATE_SOURCE_FINALIZER_STAGE",
    "US_LATE_SOURCE_EXECUTION_CONFIG_INPUT",
    "US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT",
    "US_LATE_PRIMARY_PUF_INPUT_INVENTORY",
    "US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION",
    "US_LATE_PRODUCER_REGISTRY_SCHEMA_VERSION",
    "US_LATE_PRODUCER_TRANSITION_AUTHORITY_ID",
    "US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY",
    "US_LATE_PRODUCER_TRANSITION_AUTHORITY_VERSION",
    "US_LATE_SOURCE_INPUT_INVENTORIES",
    "US_LATE_TRANSFER_INPUT_INVENTORIES",
    "US_LATE_TRANSFER_MODEL_CONFIG_INPUT",
    "US_LATE_TRANSFER_TARGET_BANK_INPUT",
    "source_producer_name",
    "transfer_producer_name",
    "us_late_producer_schedule_payload",
    "us_late_producer_schedule_receipt",
]

# v16 declares person.s_corp_income as a whole-pool primary-PUF output: its
# certified combined-source semantics are carried by partnership_income, while
# the separate S-corporation leaf is an exact-zero universe. v15 content-binds
# the complete late dual-producer ownership matrix and
# validates that it exhausts the primary/source/transfer intersection. v14
# scopes origin-exclusive raw requirements independently of their inventory
# defaults and retires whole-pool RELSHIPP/TEN/H_TENURE transfer fallbacks. v13
# declares the primary callback's optional tax-unit pass-through reads and
# binds its complete tail-control/runtime-asset surface. v12 declared every
# primary callback person read-before-write and universe-validation column and
# removed the unusable filing-status fallback. v11 bound the complete
# packaged SourceStageSpec/default surface of every
# source callback and the source finalizer's registry/exclusion/deferral
# doctrine. v10 completed the ACS PUMS earnings-universe input declaration with
# its tax-unit link, clone role, and stable lineage fallback. v9 split that
# materializer into a declared pre-primary producer. v8 added the fixed
# seed/period and operator switches consumed by every post-clone source callback.
# v7 added the primary execution configuration and every late-transfer model
# configuration/target-bank identity to the declared external-resource surface.
# Version 6 content-bound physical Frame inputs but left those callback inputs
# implicit. Receipt v4 requires exact target-origin/model-target/realized-regime
# evidence in each late-transfer group and its canonical aggregate. Receipt v3
# reconciles repeated physical evidence and scope
# cardinalities across each execution row, binds source-receipt outputs to the
# callback receipt, and requires the primary callback to report the exact
# resources it consumed. Receipt v2 introduced exact virtual-resource payloads.
US_LATE_PRODUCER_REGISTRY_SCHEMA_VERSION = 16
US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION = 4
US_LATE_PRODUCER_TRANSITION_AUTHORITY_VERSION = 1
US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY = "us_late_producer_transition_authority"
US_LATE_PRODUCER_TRANSITION_AUTHORITY_ID = "us_stacked_late_producer_transition"
US_LATE_PRIMARY_PUF_STAGE = "primary_puf_qrf"
US_LATE_ACS_EARNINGS_UNIVERSE_STAGE = "acs_pums_earnings_universe"
US_LATE_SOURCE_FINALIZER_STAGE = "source_finalizer"
US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT = (
    "@acs_pums_earnings_universe_execution_config"
)
US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT = "@acs_pums_earnings_universe_application"
US_LATE_SOURCE_EXECUTION_CONFIG_INPUT = "@post_clone_source_execution_config"
US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT = "@source_finalizer_execution_config"
US_LATE_EXTERNAL_STAGES: tuple[str, ...] = ("post_clone_input_surface",)

_ASEC_SOURCE_SCOPE = "asec_source"
_ACS_SOURCE_SCOPE = "acs_source"
_PUF_CLONE_SCOPE = "puf_clone"
_WHOLE_POOL_SCOPE = "whole_pool"
_DEFAULT_MAX_TARGETS_PER_FIT = 8
_QUALIFIED_TUITION = "qualified_tuition_expenses"
_SSTB_EARNED_INCOME = "sstb_self_employment_income_before_lsr"
_CHILDCARE_OUTPUT = "spm_unit_pre_subsidy_childcare_expenses"
_PREGNANCY_OUTPUT = "is_pregnant"
_ADULT_CARE_ROLE_INPUT = "tax_unit_role_input"
_CLONE_ATTACHMENT_OUTPUT = "person_support_clone_index"
_SOURCE_RECEIPT_PREFIX = "@source_receipt:"
US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT = "@primary_puf_execution_config"
US_LATE_TRANSFER_MODEL_CONFIG_INPUT = "@late_transfer_model_config"
US_LATE_TRANSFER_TARGET_BANK_INPUT = "@late_transfer_target_bank"
_STRUCTURAL_ENTITIES = (
    "person",
    "household",
    "tax_unit",
    "spm_unit",
    "family",
    "marital_unit",
)
_GROUP_ENTITIES = _STRUCTURAL_ENTITIES[1:]
_ASSEMBLY_MANIFEST_INPUT = "@us_spine_assembly_manifest"
_STACKED_MANIFEST_INPUT = "@us_stacked_spine_manifest"
_PUF_ATTACHMENT_MANIFEST_INPUT = "@us_puf_clone_attachment_manifest"
_STRING_LATE_TARGETS = frozenset({"ssn_card_type", "immigration_status_str"})
_BOOLEAN_LATE_TARGETS = frozenset(
    {
        "is_incapable_of_self_care",
        "is_pregnant",
        "estate_income_would_be_qualified",
        "farm_operations_income_would_be_qualified",
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
        "sstb_self_employment_income_would_be_qualified",
        "business_is_sstb",
        "attends_eligible_educational_institution_for_american_opportunity_credit",
        "has_american_opportunity_credit_1098_t_or_exception",
        "has_american_opportunity_credit_institution_ein",
        "is_enrolled_at_least_half_time_for_american_opportunity_credit",
        "is_pursuing_credential_for_american_opportunity_credit",
        "takes_up_medicare_if_eligible",
        "would_claim_wic",
    }
)


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _late_target_value_kind(column: str) -> str:
    if column in _BOOLEAN_LATE_TARGETS or column in _STRING_LATE_TARGETS:
        return "non_null"
    return "finite_numeric"


ScopedInput = ProducerInputColumn


@dataclass(frozen=True)
class EffectiveInputRequirement:
    """One logical source input, expressed as one or more all-of alternatives."""

    label: str
    alternatives: tuple[tuple[ScopedInput, ...], ...]
    optional: bool = False
    required_scope: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.label, label="EffectiveInputRequirement.label")
        alternatives = tuple(tuple(option) for option in self.alternatives)
        if not alternatives or any(not option for option in alternatives):
            raise ValueError(
                f"Effective input {self.label!r} requires nonempty alternatives."
            )
        if any(
            not isinstance(item, ScopedInput)
            for option in alternatives
            for item in option
        ):
            raise TypeError(
                f"Effective input {self.label!r} alternatives require "
                "ScopedInput values."
            )
        canonical = tuple(
            sorted(
                (tuple(sorted(set(option))) for option in alternatives),
                key=lambda option: tuple(
                    (item.entity, item.column, item.value_kind) for item in option
                ),
            )
        )
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"Effective input {self.label!r} repeats an alternative.")
        object.__setattr__(self, "alternatives", canonical)
        if self.required_scope is not None:
            _nonempty(
                self.required_scope,
                label=f"Effective input {self.label!r} required_scope",
            )


@dataclass(frozen=True)
class SourceInputInventory:
    """Complete effective input declaration for one post-clone source kernel."""

    operator: str
    requirements: tuple[EffectiveInputRequirement, ...]

    def __post_init__(self) -> None:
        _nonempty(self.operator, label="SourceInputInventory.operator")
        requirements = tuple(self.requirements)
        if not requirements:
            raise ValueError(
                f"Source input inventory {self.operator!r} must not be empty."
            )
        if any(
            not isinstance(item, EffectiveInputRequirement) for item in requirements
        ):
            raise TypeError(
                f"Source input inventory {self.operator!r} requires "
                "EffectiveInputRequirement values."
            )
        labels = [item.label for item in requirements]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Source input inventory {self.operator!r} repeats labels: {labels}."
            )
        object.__setattr__(
            self,
            "requirements",
            tuple(sorted(requirements, key=lambda item: item.label)),
        )


@dataclass(frozen=True)
class TransferProducerGroup:
    """One canonical bounded ACS-transfer family represented by a DAG node."""

    name: str
    entity: str
    family: str
    targets: tuple[str, ...]
    target_families: TargetFamilies

    def __post_init__(self) -> None:
        _nonempty(self.name, label="TransferProducerGroup.name")
        _nonempty(self.entity, label="TransferProducerGroup.entity")
        _nonempty(self.family, label="TransferProducerGroup.family")
        targets = tuple(self.targets)
        if not targets or any(
            not isinstance(item, str) or not item for item in targets
        ):
            raise ValueError(
                f"Transfer producer group {self.name!r} requires named targets."
            )
        if len(set(targets)) != len(targets):
            raise ValueError(f"Transfer producer group {self.name!r} repeats targets.")
        expected = {self.entity: {self.family: targets}}
        materialized = {
            entity: {family: tuple(columns) for family, columns in families.items()}
            for entity, families in self.target_families.items()
        }
        if materialized != expected:
            raise ValueError(
                f"Transfer producer group {self.name!r} target_families drifted; "
                f"expected {expected}, got {materialized}."
            )
        object.__setattr__(self, "targets", targets)


def source_producer_name(operator: str) -> str:
    """Return the graph node name for one post-clone source operator."""

    return f"source:{_nonempty(operator, label='source operator')}"


def transfer_producer_name(entity: str, family: str) -> str:
    """Return the graph node name for one bounded late-transfer family."""

    return (
        f"transfer:{_nonempty(entity, label='transfer entity')}/"
        f"{_nonempty(family, label='transfer family')}"
    )


def _column(
    entity: str,
    column: str,
    *,
    value_kind: str = "non_null",
) -> ScopedInput:
    return ScopedInput(entity, column, value_kind)


def _requirement(
    label: str,
    *alternatives: Sequence[ScopedInput],
    optional: bool = False,
    required_scope: str | None = None,
) -> EffectiveInputRequirement:
    return EffectiveInputRequirement(
        label=label,
        alternatives=tuple(tuple(option) for option in alternatives),
        optional=optional,
        required_scope=required_scope,
    )


def _single(
    label: str,
    entity: str,
    column: str,
    *,
    optional: bool = False,
    value_kind: str = "non_null",
    required_scope: str | None = None,
) -> EffectiveInputRequirement:
    return _requirement(
        label,
        (_column(entity, column, value_kind=value_kind),),
        optional=optional,
        required_scope=required_scope,
    )


def _cross_grain_validation_requirements() -> tuple[EffectiveInputRequirement, ...]:
    """Return the physical and metadata surface read by stacked validators."""

    requirements: list[EffectiveInputRequirement] = []
    for entity in _STRUCTURAL_ENTITIES:
        requirements.extend(
            (
                _single(
                    f"validated_structure:{entity}_support_channel",
                    entity,
                    f"{entity}_support_channel",
                ),
                _single(
                    f"validated_structure:{entity}_support_clone_index",
                    entity,
                    f"{entity}_support_clone_index",
                    value_kind="finite_numeric",
                ),
            )
        )
    for entity in _GROUP_ENTITIES:
        requirements.extend(
            (
                _single(
                    f"validated_structure:person_{entity}_id",
                    "person",
                    f"person_{entity}_id",
                    value_kind="finite_numeric",
                ),
                _single(
                    f"validated_structure:{entity}_id",
                    entity,
                    f"{entity}_id",
                    value_kind="finite_numeric",
                ),
            )
        )
    requirements.extend(
        (
            _single(
                "validated_structure:person_id",
                "person",
                "person_id",
                value_kind="finite_numeric",
            ),
            _single(
                "validated_structure:person_spine_source_id",
                "person",
                "person_spine_source_id",
                value_kind="finite_numeric",
            ),
            _single(
                "validated_structure:person_source_id",
                "person",
                "person_source_id",
                value_kind="finite_numeric",
            ),
            _single(
                "validated_structure:household_spine_source_id",
                "household",
                "household_spine_source_id",
                value_kind="finite_numeric",
            ),
            _single(
                "validated_structure:household_source_id",
                "household",
                "household_source_id",
                value_kind="finite_numeric",
            ),
            _single(
                "validated_structure:TYPEHUGQ",
                "household",
                "TYPEHUGQ",
                value_kind="finite_numeric",
                required_scope=_ACS_SOURCE_SCOPE,
            ),
            _single(
                "validated_structure:resolved_household_weight",
                "household",
                "@resolved_weight",
            ),
            _single(
                "validated_structure:assembly_manifest",
                "frame",
                _ASSEMBLY_MANIFEST_INPUT,
            ),
            _single(
                "validated_structure:stacked_manifest",
                "frame",
                _STACKED_MANIFEST_INPUT,
            ),
            _single(
                "validated_structure:puf_attachment_manifest",
                "frame",
                _PUF_ATTACHMENT_MANIFEST_INPUT,
            ),
        )
    )
    return tuple(requirements)


_CROSS_GRAIN_VALIDATION_REQUIREMENTS = _cross_grain_validation_requirements()


_POST_CLONE_SOURCE_WRAPPER_REQUIREMENTS = (
    _single(
        "source_wrapper:execution_config",
        "person",
        US_LATE_SOURCE_EXECUTION_CONFIG_INPUT,
    ),
    _single(
        "source_wrapper:assembly_manifest",
        "frame",
        _ASSEMBLY_MANIFEST_INPUT,
    ),
    _single(
        "source_wrapper:source_evidence",
        "person",
        "PERIDNUM",
    ),
    _single(
        "source_wrapper:person_support_clone_index",
        "person",
        "person_support_clone_index",
        value_kind="finite_numeric",
    ),
    *tuple(
        _single(
            f"source_wrapper:{entity}_id",
            entity,
            f"{entity}_id",
            value_kind="finite_numeric",
        )
        for entity in _STRUCTURAL_ENTITIES
    ),
    *tuple(
        _single(
            f"source_wrapper:person_{entity}_id",
            "person",
            f"person_{entity}_id",
            value_kind="finite_numeric",
        )
        for entity in _GROUP_ENTITIES
    ),
    _single(
        "source_wrapper:resolved_household_weight",
        "household",
        "@resolved_weight",
    ),
)


_COMMON_ROLE_AWARE_INPUTS = (
    _single(
        "person_id",
        "person",
        "person_id",
        value_kind="finite_numeric",
    ),
    _single("resolved_person_weight", "person", "@resolved_weight"),
    _single("support_channel", "person", "person_support_channel"),
    _single(
        "support_clone_index",
        "person",
        "person_support_clone_index",
        optional=True,
        value_kind="finite_numeric",
    ),
    _requirement(
        "age",
        (_column("person", "age", value_kind="finite_numeric"),),
        (_column("person", "A_AGE", value_kind="finite_numeric"),),
    ),
    _requirement(
        "sex",
        (_column("person", "is_male", value_kind="finite_numeric"),),
        (_column("person", "is_female", value_kind="finite_numeric"),),
        (_column("person", "A_SEX", value_kind="finite_numeric"),),
    ),
    _single(
        "employer_health_coverage",
        "person",
        "has_esi",
        value_kind="finite_numeric",
    ),
    _single(
        "person_tax_unit_link",
        "person",
        "person_tax_unit_id",
        value_kind="finite_numeric",
    ),
    _single("tax_unit_role", "person", "tax_unit_role_input"),
    _requirement(
        "employment_income",
        (
            _column(
                "person",
                "employment_income_before_lsr",
                value_kind="finite_numeric",
            ),
        ),
        (_column("person", "WSAL_VAL", value_kind="finite_numeric"),),
    ),
    _requirement(
        "self_employment_income",
        (
            _column(
                "person",
                "self_employment_income_before_lsr",
                value_kind="finite_numeric",
            ),
        ),
        (_column("person", "SEMP_VAL", value_kind="finite_numeric"),),
    ),
    _requirement(
        "social_security_income",
        (
            _column(
                "person", "social_security_retirement", value_kind="finite_numeric"
            ),
            _column(
                "person", "social_security_disability", value_kind="finite_numeric"
            ),
            _column("person", "social_security_survivors", value_kind="finite_numeric"),
            _column(
                "person", "social_security_dependents", value_kind="finite_numeric"
            ),
        ),
        (_column("person", "SS_VAL", value_kind="finite_numeric"),),
    ),
    _single(
        "tax_unit_id",
        "tax_unit",
        "tax_unit_id",
        value_kind="finite_numeric",
    ),
    _requirement(
        "filing_status",
        (_column("tax_unit", "filing_status_input"),),
        (_column("tax_unit", "filing_status"),),
    ),
)


def _raw_person_requirements(
    columns: Sequence[str],
) -> tuple[EffectiveInputRequirement, ...]:
    return tuple(
        _single(
            f"raw_person:{column}",
            "person",
            column,
            value_kind="finite_numeric",
        )
        for column in columns
    )


def _inventory(
    operator: str,
    *requirements: EffectiveInputRequirement,
) -> SourceInputInventory:
    return SourceInputInventory(operator, tuple(requirements))


# These inventories spell out what the wrappers read, including alternate
# canonical/raw spellings and inputs which are optional only because a pinned
# sidecar or stable identity fallback exists.  ``@resolved_weight`` denotes a
# typed Frame weight, not a physical table column.
_source_input_inventories = {
    "with_us_prior_year_income_inputs": _inventory(
        "with_us_prior_year_income_inputs",
        *_raw_person_requirements(
            ("source_year", "PERIDNUM", "WSAL_VAL", "SEMP_VAL", "I_ERNVAL", "I_SEVAL")
        ),
        *_COMMON_ROLE_AWARE_INPUTS,
        _single(
            "employment_income_last_year",
            "person",
            "employment_income_last_year",
            value_kind="finite_numeric",
        ),
        _single(
            "self_employment_income_last_year",
            "person",
            "self_employment_income_last_year",
            value_kind="finite_numeric",
        ),
    ),
    "with_us_medicare_take_up_input": _inventory(
        "with_us_medicare_take_up_input",
        *_raw_person_requirements(("MCARE",)),
        _single("person_id", "person", "person_id"),
        _single("resolved_person_weight", "person", "@resolved_weight"),
    ),
    "with_us_pregnancy_inputs": _inventory(
        "with_us_pregnancy_inputs",
        *_raw_person_requirements(("A_SEX", "A_AGE")),
        _single("person_id", "person", "person_id"),
        _single("resolved_person_weight", "person", "@resolved_weight"),
        _requirement(
            "stable_source_identity",
            (
                _column("person", "source_year"),
                _column("person", "source_household_id"),
                _column("person", "source_person_id"),
            ),
            (_column("person", "person_id"),),
        ),
    ),
    "with_us_wic_claim_input": _inventory(
        "with_us_wic_claim_input",
        *_raw_person_requirements(
            (
                "age",
                "is_female",
                _PREGNANCY_OUTPUT,
                "own_children_in_household",
                "person_family_id",
            )
        ),
        _single("resolved_person_weight", "person", "@resolved_weight"),
        _requirement(
            "stable_source_identity",
            (
                _column("person", "source_year"),
                _column("person", "source_household_id"),
                _column("person", "source_person_id"),
            ),
            (_column("person", "person_support_source_id"),),
            (_column("person", "person_id"),),
        ),
    ),
    "impute_us_housing_assistance_to_puf_support": _inventory(
        "impute_us_housing_assistance_to_puf_support",
        *_COMMON_ROLE_AWARE_INPUTS,
        _single("person_spm_unit_link", "person", "person_spm_unit_id"),
        _single("spm_unit_id", "spm_unit", "spm_unit_id"),
        _single(
            "housing_assistance_receipt",
            "spm_unit",
            "receives_housing_assistance",
            value_kind="finite_numeric",
        ),
        _single(
            "housing_assistance_takeup",
            "spm_unit",
            "takes_up_housing_assistance_if_eligible",
            value_kind="finite_numeric",
        ),
        _single("spm_support_channel", "spm_unit", "spm_unit_support_channel"),
        _single(
            "spm_support_clone_index",
            "spm_unit",
            "spm_unit_support_clone_index",
            optional=True,
        ),
    ),
    "with_us_child_support_inputs": _inventory(
        "with_us_child_support_inputs",
        *_raw_person_requirements(("CSP_VAL", "CHSP_VAL")),
        *_COMMON_ROLE_AWARE_INPUTS,
    ),
    "with_us_disability_benefits": _inventory(
        "with_us_disability_benefits",
        *_raw_person_requirements(("DIS_VAL1", "DIS_SC1", "DIS_VAL2", "DIS_SC2")),
        *_COMMON_ROLE_AWARE_INPUTS,
    ),
    "with_us_workers_compensation": _inventory(
        "with_us_workers_compensation",
        *_raw_person_requirements(("WC_VAL",)),
        *_COMMON_ROLE_AWARE_INPUTS,
    ),
    "with_us_weeks_unemployed": _inventory(
        "with_us_weeks_unemployed",
        _single(
            "source_year",
            "person",
            "source_year",
            value_kind="finite_numeric",
        ),
        _single(
            "source_identity",
            "person",
            "PERIDNUM",
            value_kind="finite_numeric",
        ),
        _requirement(
            "weeks_source",
            (_column("person", "LKWEEKS", value_kind="finite_numeric"),),
        ),
        _requirement(
            "age",
            (_column("person", "age", value_kind="finite_numeric"),),
            (_column("person", "A_AGE", value_kind="finite_numeric"),),
        ),
        _requirement(
            "sex",
            (_column("person", "is_male", value_kind="finite_numeric"),),
            (_column("person", "is_female", value_kind="finite_numeric"),),
            (_column("person", "A_SEX", value_kind="finite_numeric"),),
        ),
        _requirement(
            "joint_filing_status",
            (_column("person", "tax_unit_is_joint", value_kind="finite_numeric"),),
            (
                _column("person", "person_tax_unit_id"),
                _column("tax_unit", "tax_unit_id"),
                _column("tax_unit", "filing_status_input"),
            ),
            (
                _column("person", "person_tax_unit_id"),
                _column("tax_unit", "tax_unit_id"),
                _column("tax_unit", "filing_status"),
            ),
        ),
        _requirement(
            "explicit_tax_unit_roles",
            (_column("person", "tax_unit_role_input"),),
            (
                _column("person", "is_tax_unit_head", value_kind="finite_numeric"),
                _column("person", "is_tax_unit_spouse", value_kind="finite_numeric"),
                _column("person", "is_tax_unit_dependent", value_kind="finite_numeric"),
            ),
        ),
        _requirement(
            "unemployment_compensation_predictor",
            (
                _column(
                    "person",
                    "unemployment_compensation",
                    value_kind="finite_numeric",
                ),
            ),
            (_column("person", "UC_VAL", value_kind="finite_numeric"),),
            optional=True,
        ),
        _single("support_channel", "person", "person_support_channel"),
        _single("resolved_person_weight", "person", "@resolved_weight"),
    ),
    "with_us_childcare_inputs": _inventory(
        "with_us_childcare_inputs",
        *_raw_person_requirements(("person_spm_unit_id", "SPM_CHILDCAREXPNS")),
        *_COMMON_ROLE_AWARE_INPUTS,
        _single("spm_unit_id", "spm_unit", "spm_unit_id"),
    ),
    "with_us_adult_care_inputs": _inventory(
        "with_us_adult_care_inputs",
        _single(
            "raw_person:PEDISDRS",
            "person",
            "PEDISDRS",
            value_kind="finite_numeric",
        ),
        _single(
            "raw_person:is_full_time_college_student",
            "person",
            "is_full_time_college_student",
            value_kind="finite_numeric",
        ),
        _single("age", "person", "age", value_kind="finite_numeric"),
        _single(
            "employment_income",
            "person",
            "employment_income_before_lsr",
            value_kind="finite_numeric",
        ),
        _single(
            "self_employment_income",
            "person",
            "self_employment_income_before_lsr",
            value_kind="finite_numeric",
        ),
        _single(
            "sstb_earned_income",
            "person",
            _SSTB_EARNED_INCOME,
            value_kind="finite_numeric",
        ),
        _single("tax_unit_role", "person", "tax_unit_role_input"),
        _single(
            "person_tax_unit_link",
            "person",
            "person_tax_unit_id",
            value_kind="finite_numeric",
        ),
        _single(
            "person_spm_unit_link",
            "person",
            "person_spm_unit_id",
            value_kind="finite_numeric",
        ),
        _single(
            "person_id",
            "person",
            "person_id",
            value_kind="finite_numeric",
        ),
        _requirement(
            "support_role",
            (
                _column(
                    "person",
                    "person_support_clone_index",
                    value_kind="finite_numeric",
                ),
                _column("person", "person_support_channel"),
            ),
        ),
        _single(
            "childcare_expenses",
            "spm_unit",
            _CHILDCARE_OUTPUT,
            value_kind="finite_numeric",
        ),
        _single(
            "spm_unit_id",
            "spm_unit",
            "spm_unit_id",
            value_kind="finite_numeric",
        ),
        _single(
            "tax_unit_id",
            "tax_unit",
            "tax_unit_id",
            value_kind="finite_numeric",
        ),
        _single("resolved_person_weight", "person", "@resolved_weight"),
        _single("resolved_tax_unit_weight", "tax_unit", "@resolved_weight"),
        _single("resolved_spm_unit_weight", "spm_unit", "@resolved_weight"),
    ),
    "with_us_energy_subsidy_input": _inventory(
        "with_us_energy_subsidy_input",
        *_raw_person_requirements(("person_spm_unit_id", "SPM_ENGVAL")),
        *_COMMON_ROLE_AWARE_INPUTS,
        _single("spm_unit_id", "spm_unit", "spm_unit_id"),
    ),
    "with_us_retirement_contribution_inputs": _inventory(
        "with_us_retirement_contribution_inputs",
        *_raw_person_requirements(("RETCB_VAL", "WSAL_VAL", "SEMP_VAL")),
        *_COMMON_ROLE_AWARE_INPUTS,
    ),
    "with_us_retirement_distribution_inputs": _inventory(
        "with_us_retirement_distribution_inputs",
        *_raw_person_requirements(
            (
                "DST_SC1",
                "DST_VAL1",
                "DST_SC2",
                "DST_VAL2",
                "DST_SC1_YNG",
                "DST_VAL1_YNG",
                "DST_SC2_YNG",
                "DST_VAL2_YNG",
            )
        ),
        *_COMMON_ROLE_AWARE_INPUTS,
        _single(
            "puf_taxable_ira_distribution",
            "person",
            "taxable_ira_distributions",
            value_kind="finite_numeric",
        ),
    ),
    "with_us_immigration_inputs": _inventory(
        "with_us_immigration_inputs",
        *_raw_person_requirements(
            (
                "PRCITSHP",
                "PEINUSYR",
                "PENATVTY",
                "A_AGE",
                "A_MARITL",
                "A_SPOUSE",
                "A_HSCOL",
                "WSAL_VAL",
                "SEMP_VAL",
                "MCARE",
                "CAID",
                "IHSFLG",
                "CHAMPVA",
                "MIL",
                "PEN_SC1",
                "PEN_SC2",
                "RESNSS1",
                "RESNSS2",
                "SS_YN",
                "SSI_YN",
                "PEIO1COW",
                "A_MJOCC",
                "PEAFEVER",
                "SPM_CAPHOUSESUB",
            )
        ),
        _single("person_id", "person", "person_id"),
        _single("resolved_person_weight", "person", "@resolved_weight"),
        _requirement(
            "stable_source_identity",
            (
                _column("person", "source_year"),
                _column("person", "source_person_id"),
            ),
            (_column("person", "person_id"),),
        ),
    ),
    "with_us_education_inputs": _inventory(
        "with_us_education_inputs",
        _requirement(
            "education_source",
            (_column("person", "ED_VAL", value_kind="finite_numeric"),),
        ),
        _single(
            "qualified_tuition",
            "person",
            _QUALIFIED_TUITION,
            value_kind="finite_numeric",
        ),
        _single("person_id", "person", "person_id"),
        _single("resolved_person_weight", "person", "@resolved_weight"),
    ),
}

_source_input_inventories = {
    operator: SourceInputInventory(
        operator,
        (*inventory.requirements, *_POST_CLONE_SOURCE_WRAPPER_REQUIREMENTS),
    )
    for operator, inventory in _source_input_inventories.items()
}

if set(_source_input_inventories) != set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER):
    raise RuntimeError(
        "US late source input inventories must cover the exact post-clone "
        "operator registry; "
        f"missing={sorted(set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER) - set(_source_input_inventories))}, "
        f"extra={sorted(set(_source_input_inventories) - set(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER))}."
    )
US_LATE_SOURCE_INPUT_INVENTORIES: Mapping[str, SourceInputInventory] = MappingProxyType(
    dict(_source_input_inventories)
)


US_LATE_PRIMARY_PUF_INPUT_INVENTORY = _inventory(
    US_LATE_PRIMARY_PUF_STAGE,
    *(
        requirement
        for requirement in _CROSS_GRAIN_VALIDATION_REQUIREMENTS
        if requirement.label != "validated_structure:puf_attachment_manifest"
    ),
    _single("filing_status", "tax_unit", "filing_status_input"),
    _single("age", "person", "age", value_kind="finite_numeric"),
    _requirement(
        "tax_unit_person_count",
        (
            _column("person", "person_tax_unit_id"),
            _column("tax_unit", "tax_unit_id"),
        ),
    ),
    _single(
        "employment_income",
        "person",
        "employment_income_before_lsr",
        value_kind="finite_numeric",
    ),
    _single(
        "self_employment_income",
        "person",
        "self_employment_income_before_lsr",
        value_kind="finite_numeric",
    ),
    _single(
        "taxable_interest_income",
        "person",
        "taxable_interest_income",
        value_kind="finite_numeric",
    ),
    _requirement(
        "dividend_income",
        (_column("person", "dividend_income", value_kind="finite_numeric"),),
        (
            _column("person", "qualified_dividend_income", value_kind="finite_numeric"),
            _column(
                "person",
                "non_qualified_dividend_income",
                value_kind="finite_numeric",
            ),
        ),
        (_column("tax_unit", "dividend_income", value_kind="finite_numeric"),),
    ),
    _requirement(
        "short_term_capital_gains",
        (_column("person", "short_term_capital_gains", value_kind="finite_numeric"),),
        (_column("tax_unit", "short_term_capital_gains", value_kind="finite_numeric"),),
    ),
    _requirement(
        "long_term_capital_gains",
        (
            _column(
                "person",
                "long_term_capital_gains_before_response",
                value_kind="finite_numeric",
            ),
        ),
        (_column("person", "long_term_capital_gains", value_kind="finite_numeric"),),
        (_column("tax_unit", "long_term_capital_gains", value_kind="finite_numeric"),),
    ),
    _single("person_id", "person", "person_id"),
    _single("tax_unit_id", "tax_unit", "tax_unit_id"),
    _single("support_channel", "person", "person_support_channel"),
    _single("support_clone_index", "person", "person_support_clone_index"),
    _single("resolved_tax_unit_weight", "tax_unit", "@resolved_weight"),
    *(
        _single(
            f"person_output_allocation_basis:{column}",
            "person",
            column,
            optional=True,
            value_kind="finite_numeric",
        )
        for column in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    ),
    *(
        _single(
            f"tax_unit_output_passthrough:{column}",
            "tax_unit",
            column,
            optional=True,
            value_kind="finite_numeric",
        )
        for column in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    ),
    _single(
        "qualified_tuition_allocation_fallback",
        "person",
        "is_full_time_college_student",
        optional=True,
        value_kind="finite_numeric",
    ),
    _single("puf_donor", "tax_unit", "@puf_donor_tax_units"),
    _single("primary_qrf_bank", "tax_unit", "@primary_qrf_checkpoint"),
    _single(
        "primary_puf_execution_config",
        "tax_unit",
        US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT,
    ),
)


US_LATE_ACS_EARNINGS_UNIVERSE_INPUT_INVENTORY = _inventory(
    US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
    _single("age", "person", "age", value_kind="finite_numeric"),
    _single("support_channel", "person", "person_support_channel"),
    _single("person_tax_unit_link", "person", "person_tax_unit_id"),
    _single(
        "support_clone_index",
        "person",
        "person_support_clone_index",
        value_kind="finite_numeric",
    ),
    _requirement(
        "stable_person_lineage",
        (_column("person", "person_source_id"),),
        (_column("person", "person_id"),),
    ),
    *(
        _single(
            f"raw_source:{source}",
            "person",
            source,
            value_kind="column_present",
        )
        for source in ACS_PUMS_EARNINGS_SOURCE_COLUMNS.values()
    ),
    *(
        _single(
            f"mapped_earnings:{mapped}",
            "person",
            mapped,
            optional=True,
            value_kind="finite_numeric",
        )
        for mapped in ACS_PUMS_EARNINGS_SOURCE_COLUMNS
    ),
    _single(
        "execution_config",
        "person",
        US_LATE_ACS_EARNINGS_UNIVERSE_CONFIG_INPUT,
    ),
)


def _transfer_input_inventory(group: TransferProducerGroup) -> SourceInputInventory:
    structural = [
        *_CROSS_GRAIN_VALIDATION_REQUIREMENTS,
        _single("resolved_person_weight", "person", "@resolved_weight"),
        _single("resolved_target_weight", group.entity, "@resolved_weight"),
        _single(
            "late_transfer_model_config",
            group.entity,
            US_LATE_TRANSFER_MODEL_CONFIG_INPUT,
        ),
        _single(
            "late_transfer_target_bank",
            group.entity,
            US_LATE_TRANSFER_TARGET_BANK_INPUT,
        ),
    ]
    post_transfer_structure: list[EffectiveInputRequirement] = []
    if group.name == transfer_producer_name("person", "adult_care"):
        post_transfer_structure.append(
            _single(
                "adult_care_tax_unit_role",
                "person",
                _ADULT_CARE_ROLE_INPUT,
            )
        )
    return _inventory(
        group.name,
        *structural,
        *post_transfer_structure,
        _single("age", "person", "age", value_kind="finite_numeric"),
        _single(
            "is_female",
            "person",
            "is_female",
            value_kind="finite_numeric",
        ),
        _requirement(
            "state_fips",
            (_column("person", "state_fips", value_kind="finite_numeric"),),
            (
                _column("person", "person_household_id", value_kind="finite_numeric"),
                _column("household", "household_id", value_kind="finite_numeric"),
                _column("household", "state_fips", value_kind="finite_numeric"),
            ),
        ),
        _single(
            "optional_employment_income",
            "person",
            "employment_income_before_lsr",
            optional=True,
            value_kind="finite_numeric",
        ),
        _single(
            "optional_self_employment_income",
            "person",
            "self_employment_income_before_lsr",
            optional=True,
            value_kind="finite_numeric",
        ),
        _requirement(
            "optional_social_security_income",
            (
                _column(
                    "person", "social_security_retirement", value_kind="finite_numeric"
                ),
                _column(
                    "person", "social_security_disability", value_kind="finite_numeric"
                ),
                _column(
                    "person", "social_security_dependents", value_kind="finite_numeric"
                ),
                _column(
                    "person", "social_security_survivors", value_kind="finite_numeric"
                ),
            ),
            (
                _column(
                    "person", "acs_social_security_income", value_kind="finite_numeric"
                ),
            ),
            optional=True,
        ),
        _requirement(
            "optional_retirement_income",
            (
                _column(
                    "person",
                    "taxable_private_pension_income",
                    value_kind="finite_numeric",
                ),
                _column(
                    "person",
                    "tax_exempt_private_pension_income",
                    value_kind="finite_numeric",
                ),
                _column(
                    "person", "taxable_ira_distributions", value_kind="finite_numeric"
                ),
            ),
            (_column("person", "acs_retirement_income", value_kind="finite_numeric"),),
            optional=True,
        ),
        _requirement(
            "optional_investment_income",
            (
                _column(
                    "person", "taxable_interest_income", value_kind="finite_numeric"
                ),
                _column(
                    "person", "tax_exempt_interest_income", value_kind="finite_numeric"
                ),
                _column(
                    "person", "qualified_dividend_income", value_kind="finite_numeric"
                ),
                _column(
                    "person",
                    "non_qualified_dividend_income",
                    value_kind="finite_numeric",
                ),
                _column("person", "rental_income", value_kind="finite_numeric"),
                _column("person", "estate_income", value_kind="finite_numeric"),
            ),
            (
                _column(
                    "person",
                    "acs_interest_dividend_rental_income",
                    value_kind="finite_numeric",
                ),
            ),
            optional=True,
        ),
        _requirement(
            "optional_household_head",
            (_column("person", "is_household_head", value_kind="finite_numeric"),),
            (_column("person", "A_EXPRRP", value_kind="finite_numeric"),),
            (_column("person", "A_LINENO", value_kind="finite_numeric"),),
            optional=True,
        ),
        _requirement(
            "optional_tenure",
            (_column("person", "tenure_type"),),
            (_column("spm_unit", "spm_unit_tenure_type"),),
            optional=True,
        ),
    )


def _surface_rows(
    surface: TargetFamilies,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    entity_order = (
        "person",
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    )
    unknown_entities = sorted(set(surface) - set(entity_order))
    if unknown_entities:
        raise RuntimeError(
            f"US late transfer surface names unknown entities: {unknown_entities}."
        )
    return tuple(
        (entity, family, tuple(surface[entity][family]))
        for entity in entity_order
        if entity in surface
        for family in sorted(surface[entity])
        if surface[entity][family]
    )


def _bounded_transfer_groups(
    surface: TargetFamilies,
    *,
    max_targets_per_fit: int,
) -> tuple[TransferProducerGroup, ...]:
    """Mirror ACS transfer's canonical bounded-family partition."""

    if max_targets_per_fit <= 0:
        raise ValueError("US late transfer max_targets_per_fit must be positive.")
    groups: list[TransferProducerGroup] = []
    immigration_pair = ("ssn_card_type", "immigration_status_str")
    immigration_set = set(immigration_pair)
    for entity, family, targets in _surface_rows(surface):
        atoms: list[tuple[str, ...]] = []
        pair_added = False
        for target in targets:
            if target in immigration_set and immigration_set.issubset(targets):
                if not pair_added:
                    atoms.append(immigration_pair)
                    pair_added = True
                continue
            atoms.append((target,))
        batches: list[tuple[str, ...]] = []
        current: list[str] = []
        for atom in atoms:
            if current and len(current) + len(atom) > max_targets_per_fit:
                batches.append(tuple(current))
                current = []
            current.extend(atom)
        if current:
            batches.append(tuple(current))
        bounded = (
            ((family, batches[0]),)
            if len(batches) == 1
            else tuple(
                (f"{family}__batch_{position}", batch)
                for position, batch in enumerate(batches, start=1)
            )
        )
        for bounded_family, batch in bounded:
            target_families: TargetFamilies = MappingProxyType(
                {entity: MappingProxyType({bounded_family: batch})}
            )
            groups.append(
                TransferProducerGroup(
                    name=transfer_producer_name(entity, bounded_family),
                    entity=entity,
                    family=bounded_family,
                    targets=batch,
                    target_families=target_families,
                )
            )
    return tuple(groups)


CANONICAL_US_LATE_TRANSFER_GROUPS = _bounded_transfer_groups(
    pool_post_puf_transfer_target_families(),
    max_targets_per_fit=_DEFAULT_MAX_TARGETS_PER_FIT,
)
if (
    len(CANONICAL_US_LATE_TRANSFER_GROUPS) != 19
    or sum(len(group.targets) for group in CANONICAL_US_LATE_TRANSFER_GROUPS) != 70
):
    raise RuntimeError(
        "Canonical US late transfer must contain exactly 19 bounded groups "
        "and 70 ordered targets."
    )
_canonical_late_targets = {
    target for group in CANONICAL_US_LATE_TRANSFER_GROUPS for target in group.targets
}
if (
    len(_BOOLEAN_LATE_TARGETS) != 17
    or len(_STRING_LATE_TARGETS) != 2
    or not (_BOOLEAN_LATE_TARGETS | _STRING_LATE_TARGETS) <= _canonical_late_targets
    or len(_canonical_late_targets - _BOOLEAN_LATE_TARGETS - _STRING_LATE_TARGETS) != 51
):
    raise RuntimeError(
        "Canonical US late target kinds must partition 70 targets into "
        "51 numeric, 17 boolean, and 2 string inputs."
    )
US_LATE_TRANSFER_INPUT_INVENTORIES: Mapping[str, SourceInputInventory] = (
    MappingProxyType(
        {
            group.name: _transfer_input_inventory(group)
            for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        }
    )
)


def _source_outputs() -> dict[str, tuple[ProducerOutput, ...]]:
    result: dict[str, tuple[ProducerOutput, ...]] = {}
    for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER:
        family = POOL_OPERATOR_CONTRACTS[operator].family
        outputs = PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[family]
        result[operator] = tuple(
            ProducerOutput(entity, column, _ASEC_SOURCE_SCOPE)
            for entity in sorted(outputs)
            for column in sorted(
                set(outputs[entity]) - set(FORMULA_OWNED_SOURCE_COLUMNS.get(entity, ()))
            )
            if not (
                operator == "with_us_education_inputs"
                and entity == "person"
                and column == _QUALIFIED_TUITION
            )
        )
        if not result[operator]:
            raise RuntimeError(
                f"US late source producer {operator!r} owns no persisted output."
            )
    return result


CANONICAL_US_LATE_SOURCE_OUTPUTS: Mapping[str, tuple[ProducerOutput, ...]] = (
    MappingProxyType(_source_outputs())
)


def _assert_exhaustive_late_overlap_ownership() -> None:
    """Fail import unless every permitted multi-producer touch is adjudicated."""

    primary = {
        (entity, column)
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            "primary_puf_qrf"
        ].items()
        for column in columns
    }
    physical_source_writes = {
        (entity, column)
        for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            POOL_OPERATOR_CONTRACTS[operator].family
        ].items()
        for column in columns
        if column not in FORMULA_OWNED_SOURCE_COLUMNS.get(entity, ())
    }
    callback_passthroughs = {
        target
        for targets in US_LATE_SOURCE_CALLBACK_PASSTHROUGH_OUTPUTS.values()
        for target in targets
    }
    education_passthroughs = {
        ("person", column)
        for column in set(US_EDUCATION_INPUTS_OUTPUT_COLUMNS)
        - set(US_EDUCATION_INPUTS_OWNED_OUTPUT_COLUMNS)
    }
    if callback_passthroughs != education_passthroughs:
        raise RuntimeError(
            "Canonical US education callback pass-through inventory changed: "
            f"observed={sorted(education_passthroughs)}, "
            f"declared={sorted(callback_passthroughs)}."
        )
    transfer = {
        (group.entity, target)
        for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        for target in group.targets
    }
    tail_owned = {
        *(("person", column) for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS),
        *(("tax_unit", column) for column in PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS),
    }
    recipient_owned = {
        *(("person", column) for column in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS),
        *(("tax_unit", column) for column in PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS),
    } - tail_owned
    declared = set(US_LATE_OVERLAP_OWNERSHIP_TARGETS)
    observed = primary & (physical_source_writes | callback_passthroughs) & transfer
    observed &= recipient_owned
    physical_observed = primary & physical_source_writes & transfer & recipient_owned
    expected_physical = declared - {("person", _QUALIFIED_TUITION)}
    tail_overlap = (
        primary
        & (physical_source_writes | callback_passthroughs)
        & transfer
        & tail_owned
    )
    if observed != declared:
        raise RuntimeError(
            "Canonical US late overlap ownership does not exhaust the permitted "
            f"recipient-owned dual-write surface: observed={sorted(observed)}, "
            f"declared={sorted(declared)}."
        )
    if physical_observed != expected_physical:
        raise RuntimeError(
            "Canonical US late physical dual-write surface changed: "
            f"observed={sorted(physical_observed)}, "
            f"expected={sorted(expected_physical)}."
        )
    if tail_overlap:
        raise RuntimeError(
            "Canonical US late producer DAG permits an unadjudicated tail-owned "
            f"dual write: {sorted(tail_overlap)}."
        )
    canonical_source_keys = {
        (output.entity, output.column)
        for outputs in CANONICAL_US_LATE_SOURCE_OUTPUTS.values()
        for output in outputs
    }
    if not expected_physical <= canonical_source_keys or callback_passthroughs & (
        canonical_source_keys
    ):
        raise RuntimeError(
            "Canonical US late source outputs disagree with overlap write/no-op "
            "classification."
        )


_assert_exhaustive_late_overlap_ownership()


def _target_key_rows(surface: TargetFamilies) -> set[tuple[str, str]]:
    return {
        (entity, target)
        for entity, families in surface.items()
        for targets in families.values()
        for target in targets
    }


def _inventory_contract_inputs(
    node_name: str,
    inventory: SourceInputInventory,
    *,
    required_scope: str,
) -> tuple[ProducerInput, ...]:
    """Turn every effective inventory row into an executable gate input."""

    inputs: list[ProducerInput] = []
    for requirement in inventory.requirements:
        first = requirement.alternatives[0][0]
        absence_id = f"optional_input:{node_name}:{requirement.label}"
        resolved_scope = requirement.required_scope or required_scope
        inputs.append(
            ProducerInput(
                entity=first.entity,
                column=f"@effective:{requirement.label}",
                required_scope=resolved_scope,
                producing_stage=US_LATE_EXTERNAL_STAGES[0],
                tolerated_absence_receipts=(absence_id,)
                if requirement.optional
                else (),
                alternatives=tuple(
                    tuple(
                        ProducerInputColumn(
                            item.entity,
                            item.column,
                            item.value_kind,
                        )
                        for item in alternative
                    )
                    for alternative in requirement.alternatives
                ),
            )
        )
    return tuple(inputs)


def _build_registry() -> dict[str, ProducerContract]:
    late_surface = pool_post_puf_transfer_target_families()
    late_keys = _target_key_rows(late_surface)
    puf_keys = _target_key_rows(pool_post_puf_puf_producer_target_families())
    declared_primary_outputs = tuple(
        ProducerOutput(
            entity,
            column,
            (
                _WHOLE_POOL_SCOPE
                if (entity, column) == ("person", "s_corp_income")
                else _PUF_CLONE_SCOPE
            ),
        )
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            "primary_puf_qrf"
        ].items()
        for column in columns
    ) + (ProducerOutput("person", _CLONE_ATTACHMENT_OUTPUT, _WHOLE_POOL_SCOPE),)
    structural_outputs = tuple(
        ProducerOutput(
            column.entity,
            column.column,
            requirement.required_scope or _WHOLE_POOL_SCOPE,
        )
        for requirement in _CROSS_GRAIN_VALIDATION_REQUIREMENTS
        for alternative in requirement.alternatives
        for column in alternative
        if column.entity != "frame" and column.column != "@resolved_weight"
    ) + (
        ProducerOutput(
            "frame",
            _PUF_ATTACHMENT_MANIFEST_INPUT,
            _WHOLE_POOL_SCOPE,
        ),
        *(
            ProducerOutput(entity, "@resolved_weight", _WHOLE_POOL_SCOPE)
            for entity in _STRUCTURAL_ENTITIES
        ),
    )
    primary_output_by_key: dict[tuple[str, str], ProducerOutput] = {}
    for output in (*declared_primary_outputs, *structural_outputs):
        key = (output.entity, output.column)
        previous = primary_output_by_key.get(key)
        if previous is not None and previous.coverage_scope != output.coverage_scope:
            raise RuntimeError(
                f"US primary-PUF output {key} has conflicting coverage "
                f"{previous.coverage_scope!r} and {output.coverage_scope!r}."
            )
        primary_output_by_key[key] = output
    primary_outputs = tuple(primary_output_by_key.values())
    primary_keys = {(output.entity, output.column) for output in primary_outputs}
    source_owner: dict[tuple[str, str], str] = {}
    for operator, outputs in CANONICAL_US_LATE_SOURCE_OUTPUTS.items():
        for output in outputs:
            key = (output.entity, output.column)
            if key not in late_keys:
                continue
            previous = source_owner.setdefault(key, operator)
            if previous != operator:
                raise RuntimeError(
                    f"US late target {key} has source owners {previous!r} and "
                    f"{operator!r}."
                )
    # Education consumes PUF-owned tuition; it does not produce or pass it
    # through.  Keep the ownership correction fail-closed here as well as in
    # the source output declaration above.
    if ("person", _QUALIFIED_TUITION) in source_owner:
        raise RuntimeError(
            "US education late-source ownership must exclude qualified tuition."
        )

    registry: dict[str, ProducerContract] = {}
    registry[US_LATE_ACS_EARNINGS_UNIVERSE_STAGE] = ProducerContract(
        name=US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
        kind="acs_earnings_universe",
        inputs=_inventory_contract_inputs(
            US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
            US_LATE_ACS_EARNINGS_UNIVERSE_INPUT_INVENTORY,
            required_scope=_ACS_SOURCE_SCOPE,
        ),
        outputs=(
            *(
                ProducerOutput("person", mapped, _ACS_SOURCE_SCOPE)
                for mapped in ACS_PUMS_EARNINGS_SOURCE_COLUMNS
            ),
            ProducerOutput(
                "frame",
                US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT,
                _WHOLE_POOL_SCOPE,
            ),
        ),
    )
    registry[US_LATE_PRIMARY_PUF_STAGE] = ProducerContract(
        name=US_LATE_PRIMARY_PUF_STAGE,
        kind="primary_puf",
        inputs=(
            *_inventory_contract_inputs(
                US_LATE_PRIMARY_PUF_STAGE,
                US_LATE_PRIMARY_PUF_INPUT_INVENTORY,
                required_scope=_WHOLE_POOL_SCOPE,
            ),
            *(
                ProducerInput(
                    "person",
                    mapped,
                    _ACS_SOURCE_SCOPE,
                    US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
                    alternatives=(
                        (ProducerInputColumn("person", mapped, "finite_numeric"),),
                    ),
                )
                for mapped in ACS_PUMS_EARNINGS_SOURCE_COLUMNS
            ),
            ProducerInput(
                "frame",
                US_LATE_ACS_EARNINGS_UNIVERSE_RECEIPT_INPUT,
                _WHOLE_POOL_SCOPE,
                US_LATE_ACS_EARNINGS_UNIVERSE_STAGE,
            ),
            *(
                ProducerInput(
                    "person",
                    raw_source,
                    _ACS_SOURCE_SCOPE,
                    US_LATE_EXTERNAL_STAGES[0],
                    alternatives=(
                        (
                            ProducerInputColumn(
                                "person",
                                raw_source,
                                "column_present",
                            ),
                        ),
                    ),
                )
                for raw_source in ACS_PUMS_EARNINGS_SOURCE_COLUMNS.values()
            ),
        ),
        outputs=primary_outputs,
    )

    group_by_target = {
        (group.entity, target): group
        for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        for target in group.targets
    }
    source_dependencies: dict[str, list[ProducerInput]] = defaultdict(list)
    source_dependencies["with_us_wic_claim_input"].append(
        ProducerInput(
            "person",
            _PREGNANCY_OUTPUT,
            _ASEC_SOURCE_SCOPE,
            source_producer_name("with_us_pregnancy_inputs"),
            alternatives=(
                (ProducerInputColumn("person", _PREGNANCY_OUTPUT, "non_null"),),
            ),
        )
    )
    source_dependencies["with_us_adult_care_inputs"].extend(
        (
            ProducerInput(
                "spm_unit",
                _CHILDCARE_OUTPUT,
                _ASEC_SOURCE_SCOPE,
                source_producer_name("with_us_childcare_inputs"),
                alternatives=(
                    (
                        ProducerInputColumn(
                            "spm_unit",
                            _CHILDCARE_OUTPUT,
                            "finite_numeric",
                        ),
                    ),
                ),
            ),
            ProducerInput(
                "person",
                _SSTB_EARNED_INCOME,
                _ASEC_SOURCE_SCOPE,
                group_by_target[("person", _SSTB_EARNED_INCOME)].name,
                alternatives=(
                    (
                        ProducerInputColumn(
                            "person",
                            _SSTB_EARNED_INCOME,
                            "finite_numeric",
                        ),
                    ),
                ),
            ),
        )
    )
    source_dependencies["with_us_education_inputs"].append(
        ProducerInput(
            "person",
            _QUALIFIED_TUITION,
            _ASEC_SOURCE_SCOPE,
            group_by_target[("person", _QUALIFIED_TUITION)].name,
            alternatives=(
                (
                    ProducerInputColumn(
                        "person",
                        _QUALIFIED_TUITION,
                        "finite_numeric",
                    ),
                ),
            ),
        )
    )
    for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER:
        name = source_producer_name(operator)
        direct_dependencies = [
            *source_dependencies[operator],
            ProducerInput(
                "person",
                _CLONE_ATTACHMENT_OUTPUT,
                _WHOLE_POOL_SCOPE,
                US_LATE_PRIMARY_PUF_STAGE,
            ),
        ]
        direct_dependency_keys = {
            (item.entity, item.column) for item in direct_dependencies
        }
        for requirement in US_LATE_SOURCE_INPUT_INVENTORIES[operator].requirements:
            for alternative in requirement.alternatives:
                for item in alternative:
                    key = (item.entity, item.column)
                    if key in primary_keys and key not in direct_dependency_keys:
                        direct_dependencies.append(
                            ProducerInput(
                                item.entity,
                                item.column,
                                _PUF_CLONE_SCOPE,
                                US_LATE_PRIMARY_PUF_STAGE,
                            )
                        )
                        direct_dependency_keys.add(key)
        registry[name] = ProducerContract(
            name=name,
            kind="post_clone_source",
            inputs=tuple(
                {
                    *direct_dependencies,
                    *_inventory_contract_inputs(
                        name,
                        US_LATE_SOURCE_INPUT_INVENTORIES[operator],
                        required_scope=_ASEC_SOURCE_SCOPE,
                    ),
                }
            ),
            outputs=(
                *CANONICAL_US_LATE_SOURCE_OUTPUTS[operator],
                ProducerOutput(
                    "person",
                    f"{_SOURCE_RECEIPT_PREFIX}{operator}",
                    "receipt",
                ),
            ),
        )

    registry[US_LATE_SOURCE_FINALIZER_STAGE] = ProducerContract(
        name=US_LATE_SOURCE_FINALIZER_STAGE,
        kind="source_finalizer",
        inputs=(
            *(
                ProducerInput(
                    "person",
                    f"{_SOURCE_RECEIPT_PREFIX}{operator}",
                    _WHOLE_POOL_SCOPE,
                    source_producer_name(operator),
                )
                for operator in POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
            ),
            ProducerInput(
                "person",
                US_LATE_SOURCE_FINALIZER_EXECUTION_CONFIG_INPUT,
                _WHOLE_POOL_SCOPE,
                US_LATE_EXTERNAL_STAGES[0],
            ),
        ),
        outputs=tuple(
            ProducerOutput(
                declaration["entity"],
                column,
                _WHOLE_POOL_SCOPE,
            )
            for column, declaration in POOL_DEFERRED_TRANSFER_INPUTS.items()
        ),
    )

    covered: set[tuple[str, str]] = set()
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        inputs: list[ProducerInput] = list(
            _inventory_contract_inputs(
                group.name,
                US_LATE_TRANSFER_INPUT_INVENTORIES[group.name],
                required_scope=_WHOLE_POOL_SCOPE,
            )
        )
        inputs.extend(
            (
                ProducerInput(
                    "person",
                    "tax_exempt_interest_income",
                    _PUF_CLONE_SCOPE,
                    US_LATE_PRIMARY_PUF_STAGE,
                    alternatives=(
                        (
                            ProducerInputColumn(
                                "person",
                                "tax_exempt_interest_income",
                                "finite_numeric",
                            ),
                        ),
                    ),
                ),
                ProducerInput(
                    "person",
                    "estate_income",
                    _PUF_CLONE_SCOPE,
                    US_LATE_PRIMARY_PUF_STAGE,
                    alternatives=(
                        (
                            ProducerInputColumn(
                                "person",
                                "estate_income",
                                "finite_numeric",
                            ),
                        ),
                    ),
                ),
            )
        )
        direct_dependency_keys = {
            (item.entity, item.column)
            for item in inputs
            if not item.column.startswith("@effective:")
        }
        for requirement in US_LATE_TRANSFER_INPUT_INVENTORIES[group.name].requirements:
            for alternative in requirement.alternatives:
                for item in alternative:
                    key = (item.entity, item.column)
                    output = primary_output_by_key.get(key)
                    if output is None or key in direct_dependency_keys:
                        continue
                    inputs.append(
                        ProducerInput(
                            item.entity,
                            item.column,
                            output.coverage_scope,
                            US_LATE_PRIMARY_PUF_STAGE,
                            alternatives=((item,),),
                        )
                    )
                    direct_dependency_keys.add(key)
        outputs: list[ProducerOutput] = []
        for target in group.targets:
            key = (group.entity, target)
            covered.add(key)
            if key in puf_keys:
                inputs.append(
                    ProducerInput(
                        group.entity,
                        target,
                        _PUF_CLONE_SCOPE,
                        US_LATE_PRIMARY_PUF_STAGE,
                        alternatives=(
                            (
                                ProducerInputColumn(
                                    group.entity,
                                    target,
                                    _late_target_value_kind(target),
                                ),
                            ),
                        ),
                    )
                )
            owner = source_owner.get(key)
            if owner is not None:
                inputs.append(
                    ProducerInput(
                        group.entity,
                        target,
                        _ASEC_SOURCE_SCOPE,
                        source_producer_name(owner),
                        alternatives=(
                            (
                                ProducerInputColumn(
                                    group.entity,
                                    target,
                                    _late_target_value_kind(target),
                                ),
                            ),
                        ),
                    )
                )
            if key not in puf_keys and owner is None:
                raise RuntimeError(
                    f"US late transfer target {key} has no declared producer."
                )
            outputs.append(ProducerOutput(group.entity, target, _WHOLE_POOL_SCOPE))
        registry[group.name] = ProducerContract(
            name=group.name,
            kind="late_transfer",
            inputs=tuple(set(inputs)),
            outputs=tuple(outputs),
        )
    if covered != late_keys:
        raise RuntimeError(
            "US late transfer groups do not exactly cover the canonical surface; "
            f"missing={sorted(late_keys - covered)}, extra={sorted(covered - late_keys)}."
        )
    return registry


CANONICAL_US_LATE_PRODUCER_REGISTRY: Mapping[str, ProducerContract] = MappingProxyType(
    _build_registry()
)
CANONICAL_US_LATE_PRODUCER_SCHEDULE: ProducerSchedule = derive_producer_schedule(
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    external_stages=US_LATE_EXTERNAL_STAGES,
)


def _inventory_payload(inventory: SourceInputInventory) -> dict[str, object]:
    return {
        "operator": inventory.operator,
        "requirements": [
            {
                "label": requirement.label,
                "optional": requirement.optional,
                "required_scope": requirement.required_scope,
                "alternatives": [
                    [
                        {
                            "entity": item.entity,
                            "column": item.column,
                            "value_kind": item.value_kind,
                        }
                        for item in alternative
                    ]
                    for alternative in requirement.alternatives
                ],
            }
            for requirement in inventory.requirements
        ],
    }


def us_late_producer_schedule_payload() -> dict[str, object]:
    """Return the complete JSON-safe declaration bound into checkpoint identity."""

    schedule = CANONICAL_US_LATE_PRODUCER_SCHEDULE
    return {
        "schema_version": US_LATE_PRODUCER_REGISTRY_SCHEMA_VERSION,
        "overlap_ownership": dict(us_late_overlap_ownership_receipt()),
        "execution_receipt_contract": {
            "version": US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
            "row_binding": (
                "declared_globally_reconciled_input_and_scope_exact_output_"
                "source_and_primary_callback_resource_receipt_and_previous_"
                "execution_sha256"
            ),
            "virtual_resource_binding": (
                "exact_kind_specific_semantic_payload_and_sha256"
            ),
            "top_binding": (
                "entry_and_output_frame_sha256_execution_chain_source_"
                "completion_and_nineteen_transfer_groups_with_exact_target_"
                "origin_regime_evidence"
            ),
            "transition_authority": {
                "authority_id": US_LATE_PRODUCER_TRANSITION_AUTHORITY_ID,
                "metadata_key": US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY,
                "version": US_LATE_PRODUCER_TRANSITION_AUTHORITY_VERSION,
                "independent_digest_required": True,
            },
        },
        "schedule_sha256": schedule.sha256,
        "external_stages": list(US_LATE_EXTERNAL_STAGES),
        "order": list(schedule.order),
        "waves": [list(wave) for wave in schedule.waves],
        "edges": [list(edge) for edge in schedule.edges],
        "transfer_groups": [
            {
                "name": group.name,
                "entity": group.entity,
                "family": group.family,
                "targets": list(group.targets),
            }
            for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        ],
        "source_input_inventories": [
            _inventory_payload(US_LATE_SOURCE_INPUT_INVENTORIES[operator])
            for operator in sorted(US_LATE_SOURCE_INPUT_INVENTORIES)
        ],
        "primary_puf_input_inventory": _inventory_payload(
            US_LATE_PRIMARY_PUF_INPUT_INVENTORY
        ),
        "acs_earnings_universe_input_inventory": _inventory_payload(
            US_LATE_ACS_EARNINGS_UNIVERSE_INPUT_INVENTORY
        ),
        "transfer_input_inventories": [
            _inventory_payload(US_LATE_TRANSFER_INPUT_INVENTORIES[name])
            for name in sorted(US_LATE_TRANSFER_INPUT_INVENTORIES)
        ],
    }


def us_late_producer_schedule_receipt() -> Mapping[str, object]:
    """Return the byte-stable production schedule identity and useful counts."""

    payload = us_late_producer_schedule_payload()
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return MappingProxyType(
        {
            **payload,
            "payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "producer_count": len(CANONICAL_US_LATE_PRODUCER_REGISTRY),
            "source_producer_count": len(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER),
            "transfer_group_count": len(CANONICAL_US_LATE_TRANSFER_GROUPS),
            "transfer_target_count": sum(
                len(group.targets) for group in CANONICAL_US_LATE_TRANSFER_GROUPS
            ),
            "status": "derived_and_import_validated",
        }
    )
