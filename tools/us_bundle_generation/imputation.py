"""Extract the US imputation specification from the constants-era authority.

This module is a migration boundary, not a second runtime authority.  The
one-shot US bundle generator calls :func:`build_imputation`, writes the returned
JSON-shaped value to package data, and the spec compiler subsequently owns the
same declaration.  Keep the extraction fail-closed: a changed live constant
must either be represented deliberately here or make one of the invariants
below fail.

The late producer resource receipt contains interpreter paths, Python-version
facts, environment overrides, and CPU-count-derived worker settings.  Those
values are execution-profile state, not portable configuration.  The emitted
resource-semantics component therefore carries an exact structured resolution
template for those fields instead of capturing this workstation's values.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
from microcosm.build.spec_engine.typed_closure import (
    compile_producer_outputs,
    producer_scope_registry_wire,
)
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    acs_transfer_execution_contract_identity,
)
from microcosm.build.us_runtime.multispine_pool import POOL_RANDOM_SEED
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    PRIMARY_QRF_TARGET_ORDER,
    PRIMARY_QRF_TARGET_ORDER_SHA256,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)
from microcosm.build.us_runtime.stacked_spine import (
    stacked_gap_fill_plan,
    stacked_gap_fill_producer_schedule_receipt,
    stacked_late_producer_resource_semantics_receipt,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_SCHEDULE,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    us_late_producer_schedule_receipt,
)
from microcosm.fit import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)

_LEGACY_CLONE_ATTACHMENT_FRACTION = 1.0
_LEGACY_CLONE_ATTACHMENT_SEED = 578
_F_P_WAIVER = "F-P: eligibility concepts absent"

_PUF_ATTACHMENT_REF: dict[str, str] = {
    "domain": "spine",
    "support_role": "puf_tax_detail",
    "pointer": "/attachment",
}
_PUF_TAIL_SUPPORT_REF: dict[str, str] = {
    "domain": "spine",
    "support_role": "puf_tax_detail",
    "pointer": "/tail_support/legacy_contract",
}
_BUILD_MODEL_SEED_REF: dict[str, str] = {
    "domain": "seed_protocol",
    "value_source": "run_request.build_model_seed",
}
_TARGET_PERIOD_REF: dict[str, str] = {
    "domain": "bundle",
    "pointer": "/dataset_run/target_period",
}
_SOURCE_OPERATOR_REGISTRY_REF: dict[str, str] = {
    "domain": "spine",
    "pointer": "/pipeline_contract/post_clone_source_operator_order",
}

_PARTICIPATION_TARGETS = frozenset(
    {
        "has_champva_health_coverage_at_interview",
        "has_esi",
        "has_indian_health_service_coverage_at_interview",
        "has_marketplace_health_coverage_at_interview",
        "has_medicaid_health_coverage_at_interview",
        "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        "has_other_means_tested_health_coverage_at_interview",
        "has_tricare_health_coverage_at_interview",
        "has_va_health_coverage_at_interview",
        "is_tanf_enrolled",
        "receives_housing_assistance",
        "receives_snap",
        "receives_wic",
        "takes_up_housing_assistance_if_eligible",
        "takes_up_medicare_if_eligible",
        "would_claim_wic",
    }
)

# RFC v3 concept grammar is deliberately separate from the constants-era
# predictor blocks.  These are the eligibility-bearing columns a future
# family-specific predictor declaration must cover; F0 records, but does not
# silently add, those predictors.  The veteran and military concepts retain
# the approved RFC's exact golden example.
_CONCEPTS: dict[str, tuple[str, ...]] = {
    "american_indian_status": ("is_american_indian_or_alaska_native",),
    "citizenship_status": ("is_us_citizen",),
    "dependent_child_status": ("own_children_in_household",),
    "disability_status": (
        "has_hearing_difficulty",
        "has_vision_difficulty",
    ),
    "employment_attachment": (
        "hours_worked_last_week",
        "weeks_worked_last_year",
    ),
    "household_income_eligibility": (
        "spm_unit_net_income",
        "spm_unit_size",
    ),
    "housing_need": (
        "pre_subsidy_rent",
        "tenure_type",
    ),
    "medicare_coverage_context": ("acs_hins_medicare",),
    "military_coverage_context": ("acs_hins_va",),
    "pregnancy_status": ("is_pregnant",),
    "private_coverage_context": (
        "acs_hins_employer",
        "acs_hins_direct_purchase",
    ),
    "public_coverage_context": (
        "acs_hins_medicaid",
        "acs_hins_other_public",
    ),
    "veteran_status": (
        "is_veteran",
        "receives_va_payments",
    ),
}

_PARTICIPATION_TARGET_CONCEPTS: dict[str, tuple[str, ...]] = {
    "has_champva_health_coverage_at_interview": (
        "veteran_status",
        "military_coverage_context",
    ),
    "has_esi": (
        "employment_attachment",
        "private_coverage_context",
    ),
    "has_indian_health_service_coverage_at_interview": ("american_indian_status",),
    "has_marketplace_health_coverage_at_interview": (
        "citizenship_status",
        "household_income_eligibility",
        "private_coverage_context",
    ),
    "has_medicaid_health_coverage_at_interview": (
        "dependent_child_status",
        "disability_status",
        "household_income_eligibility",
        "public_coverage_context",
    ),
    "has_non_marketplace_direct_purchase_health_coverage_at_interview": (
        "household_income_eligibility",
        "private_coverage_context",
    ),
    "has_other_means_tested_health_coverage_at_interview": (
        "disability_status",
        "household_income_eligibility",
        "public_coverage_context",
    ),
    "has_tricare_health_coverage_at_interview": ("veteran_status",),
    "has_va_health_coverage_at_interview": (
        "veteran_status",
        "military_coverage_context",
    ),
    "is_tanf_enrolled": (
        "dependent_child_status",
        "household_income_eligibility",
    ),
    "receives_housing_assistance": (
        "household_income_eligibility",
        "housing_need",
    ),
    "receives_snap": (
        "dependent_child_status",
        "disability_status",
        "household_income_eligibility",
    ),
    "receives_wic": (
        "dependent_child_status",
        "household_income_eligibility",
        "pregnancy_status",
    ),
    "takes_up_housing_assistance_if_eligible": (
        "household_income_eligibility",
        "housing_need",
    ),
    "takes_up_medicare_if_eligible": (
        "disability_status",
        "medicare_coverage_context",
    ),
    "would_claim_wic": (
        "dependent_child_status",
        "pregnancy_status",
    ),
}

_INTEGER_CATEGORY_TARGETS = frozenset(
    {
        "first_home_mortgage_origination_year",
        "second_home_mortgage_origination_year",
    }
)
_INTEGER_COUNT_TARGETS = frozenset({"own_children_in_household"})
_F_P_WAIVER_ID = "f_p_eligibility_concepts_absent"


def _json_ready(value: Any) -> Any:
    """Return a fresh JSON-shaped copy of frozen live authority values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_ready(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(nested) for nested in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_ready(nested) for nested in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"US imputation authority must be JSON-shaped; got {type(value).__name__}."
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _without_sha256(value: Mapping[str, object]) -> dict[str, object]:
    """Copy a canonical receipt body without its derived digest snapshot."""

    result = deepcopy(dict(value))
    result.pop("sha256", None)
    return result


def _target(
    metadata: PolicyEngineUSVariableMetadataIndex,
    *,
    entity: str,
    name: str,
) -> dict[str, object]:
    declaration = metadata.variable_metadata(name)
    if declaration.entity != entity:
        raise RuntimeError(
            f"Imputation target {entity}.{name} is owned by "
            f"{declaration.entity!r} in PolicyEngine-US."
        )
    if declaration.dtype == "bool":
        value_kind = "flag"
    elif declaration.dtype == "float":
        value_kind = "amount"
    elif declaration.dtype == "str":
        value_kind = "category"
    elif name in _INTEGER_CATEGORY_TARGETS:
        value_kind = "category"
    elif name in _INTEGER_COUNT_TARGETS:
        value_kind = "count"
    else:
        raise RuntimeError(
            f"Integer imputation target {entity}.{name} needs an explicit "
            "amount/count/category classification."
        )
    result: dict[str, object] = {
        "name": name,
        "entity": entity,
        "value_kind": value_kind,
        "dtype": declaration.dtype,
        "period": declaration.period,
        "requires_concepts": list(_PARTICIPATION_TARGET_CONCEPTS.get(name, ())),
    }
    if name in _PARTICIPATION_TARGETS:
        result["waiver"] = _F_P_WAIVER_ID
    return result


def _canonical_output_coverage_scope(
    *,
    producer: str,
    entity: str,
    column: str,
) -> str:
    """Read a family target's row coverage from the constants-era contract.

    Family targets own modeled columns, while graph nodes own structural and
    virtual outputs.  Coverage is the one modeled-output fact that is not
    otherwise present on a target (notably ``person.s_corp_income`` is a
    whole-pool primary output while the other primary targets are clone-only),
    so the migration tool transfers it onto the target before dropping the
    duplicate graph row.
    """

    canonical = json.loads(CANONICAL_US_LATE_PRODUCER_SCHEDULE.canonical_json)
    matches = [
        output
        for contract in canonical["contracts"]
        if contract["name"] == producer
        for output in contract["outputs"]
        if output["entity"] == entity and output["column"] == column
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Family target must resolve exactly one canonical producer output: "
            f"{producer}/{entity}.{column}; matches={len(matches)}."
        )
    return str(matches[0]["coverage_scope"])


def _predictor_blocks(
    transfer_contract: Mapping[str, object],
) -> dict[str, object]:
    person_required = list(transfer_contract["person_required_predictors"])
    person_optional = list(transfer_contract["person_optional_predictors"])
    housing_required = list(transfer_contract["housing"]["mandatory_features"])
    ordinary_optional = [
        column for column in person_optional if column not in housing_required
    ]
    optional_structure = [
        column for column in person_optional if column in housing_required
    ]
    optional_names = transfer_contract["group_optional_names"]
    group_optional = [optional_names[column] for column in person_optional]
    return {
        "acs_person_required": {
            "columns": person_required,
            "tags": ["acs", "person", "required"],
            "availability": "always",
        },
        "acs_person_optional_income": {
            "columns": ordinary_optional,
            "tags": ["acs", "person", "optional", "income"],
            "availability": "observed",
        },
        "acs_person_optional_structure": {
            "columns": optional_structure,
            "tags": ["acs", "person", "optional", "structure"],
            "availability": "observed",
        },
        "acs_person_housing_required": {
            "columns": housing_required,
            "tags": ["acs", "person", "housing", "required"],
            "availability": "always",
        },
        "acs_group_required": {
            "columns": list(transfer_contract["group_required_predictors"]),
            "tags": ["acs", "group", "required"],
            "availability": "always",
        },
        "acs_group_optional": {
            "columns": group_optional,
            "tags": ["acs", "group", "optional"],
            "availability": "observed",
        },
        "puf_tax_detail": {
            "columns": list(PUF_TAX_DETAIL_DEFAULT_PREDICTORS),
            "tags": ["puf", "primary", "required"],
            "availability": "always",
        },
    }


def _acs_predictor_blocks(*, entity: str, housing: bool = False) -> list[str]:
    if entity == "person":
        return [
            "acs_person_required",
            "acs_person_optional_income",
            (
                "acs_person_housing_required"
                if housing
                else "acs_person_optional_structure"
            ),
        ]
    return ["acs_group_required", "acs_group_optional"]


def _missing_participation_concepts(
    *,
    families: Sequence[Mapping[str, object]],
    predictor_blocks: Mapping[str, object],
) -> dict[str, list[str]]:
    """Prove the F-P waiver describes today's eligibility-blind predictors."""

    block_columns = {
        block_id: frozenset(
            str(column)
            for column in _array_like(
                _mapping_like(block, f"predictor block {block_id}").get("columns"),
                f"predictor block {block_id} columns",
            )
        )
        for block_id, block in predictor_blocks.items()
    }
    missing_by_target: dict[str, tuple[str, ...]] = {}
    for family in families:
        family_id = str(family["id"])
        predictor_columns: set[str] = set()
        for block_id in _array_like(
            family["predictors"], f"family {family_id} predictors"
        ):
            block_name = str(block_id)
            try:
                predictor_columns.update(block_columns[block_name])
            except KeyError as error:
                raise RuntimeError(
                    f"US imputation family {family_id} references unknown "
                    f"predictor block {block_name!r}."
                ) from error
        for target in _array_like(family["targets"], f"family {family_id} targets"):
            target_row = _mapping_like(target, f"family {family_id} target")
            target_name = str(target_row["name"])
            required = _PARTICIPATION_TARGET_CONCEPTS.get(target_name)
            if required is None:
                continue
            missing = tuple(
                concept_id
                for concept_id in required
                if not set(_CONCEPTS[concept_id]).issubset(predictor_columns)
            )
            # This is an attestation of the constants-era gap, not permission
            # to keep a stale waiver after a predictor family is repaired.
            if missing != required:
                covered = sorted(set(required) - set(missing))
                raise RuntimeError(
                    f"US F-P waiver is stale for {target_name!r} in {family_id!r}; "
                    f"covered concepts={covered}."
                )
            prior = missing_by_target.setdefault(target_name, missing)
            if prior != missing:
                raise RuntimeError(
                    f"US F-P missing concepts differ across families for "
                    f"{target_name!r}."
                )
    if set(missing_by_target) != _PARTICIPATION_TARGETS:
        raise RuntimeError(
            "US F-P missing-concept coverage does not match participation targets."
        )
    return {
        target: list(missing_by_target[target]) for target in sorted(missing_by_target)
    }


def _mapping_like(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{location} must be a mapping.")
    return value


def _array_like(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise RuntimeError(f"{location} must be an array.")
    return value


def _ordered_direction_targets(direction: object) -> list[str]:
    return [
        target
        for families in direction.target_families.values()
        for targets in families.values()
        for target in targets
    ]


def _early_families(
    metadata: PolicyEngineUSVariableMetadataIndex,
    *,
    gap_fill_schedule: Mapping[str, object],
) -> list[dict[str, object]]:
    producer_by_target: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for direction_value in _array_like(
        gap_fill_schedule["directions"], "gap-fill producer schedule directions"
    ):
        direction_row = _mapping_like(
            direction_value, "gap-fill producer schedule direction"
        )
        direction_name = str(direction_row["name"])
        for target_value in _array_like(
            direction_row["targets"], "gap-fill producer schedule targets"
        ):
            target = _mapping_like(target_value, "gap-fill producer schedule target")
            key = (
                direction_name,
                str(target["entity"]),
                str(target["family"]),
                str(target["column"]),
            )
            if key in producer_by_target:
                raise RuntimeError(f"Gap-fill producer schedule repeats {key!r}.")
            producer_by_target[key] = {
                "operator": target["producer"],
                "order_index": target["producer_order_index"],
                "execution_scope": target["execution_scope"],
                "stage": target["producer_stage"],
            }

    families: list[dict[str, object]] = []
    consumed_producers: set[tuple[str, str, str, str]] = set()
    for direction in stacked_gap_fill_plan():
        absence_by_target: dict[tuple[str, str], list[dict[str, object]]] = {}
        for rule in direction.recipient_absence_rules:
            absence_by_target.setdefault((rule.entity, rule.column), []).append(
                _json_ready(rule)
            )
        for entity, entity_families in direction.target_families.items():
            for family_name, targets in entity_families.items():
                family_id = f"early/{direction.name}/{entity}/{family_name}"
                target_rows = [
                    _target(metadata, entity=entity, name=target) for target in targets
                ]
                for row in target_rows:
                    producer_key = (
                        direction.name,
                        entity,
                        family_name,
                        str(row["name"]),
                    )
                    try:
                        row["producer_binding"] = deepcopy(
                            producer_by_target[producer_key]
                        )
                    except KeyError as error:
                        raise RuntimeError(
                            "Gap-fill family target has no producer binding: "
                            f"{producer_key!r}."
                        ) from error
                    consumed_producers.add(producer_key)
                    rules = absence_by_target.get((entity, str(row["name"])), [])
                    if rules:
                        row["recipient_absence_rules"] = rules
                families.append(
                    {
                        "id": family_id,
                        "stage": "gap_fill_stacked_spine",
                        "entities": [entity],
                        "direction": direction.name,
                        "donor": {"channel": direction.donor_channel},
                        "donor_contract": {
                            "spine": ASEC_PUF_DONOR_SPINE,
                            "selection": "all_rows_in_declared_donor_channel",
                        },
                        "recipient": {"channel": direction.recipient_channel},
                        "recipient_contract": {
                            "selection": "all_rows_in_declared_recipient_channel",
                        },
                        "model": "regime_gated_qrf",
                        "predictors": _acs_predictor_blocks(
                            entity=entity,
                            housing=family_name == "housing",
                        ),
                        "max_targets_per_fit": (
                            DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                        ),
                        "execution_contract": "acs_transfer_early",
                        "targets": target_rows,
                    }
                )
    unconsumed = sorted(set(producer_by_target) - consumed_producers)
    if unconsumed:
        raise RuntimeError(
            "Gap-fill producer schedule has targets outside the family ledger: "
            f"{unconsumed!r}."
        )
    return families


def _primary_family(
    metadata: PolicyEngineUSVariableMetadataIndex,
) -> dict[str, object]:
    person_targets = set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
    tax_unit_targets = set(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS)
    if person_targets & tax_unit_targets:
        raise RuntimeError("Primary PUF target entities overlap.")
    if person_targets | tax_unit_targets != set(PRIMARY_QRF_TARGET_ORDER):
        raise RuntimeError("Primary PUF target entity surfaces do not cover the chain.")

    target_rows: list[dict[str, object]] = []
    for target_name in PRIMARY_QRF_TARGET_ORDER:
        entity = "person" if target_name in person_targets else "tax_unit"
        target = _target(metadata, entity=entity, name=target_name)
        target["output_coverage_scope"] = _canonical_output_coverage_scope(
            producer="primary_puf_qrf",
            entity=entity,
            column=target_name,
        )
        target_rows.append(target)

    return {
        "id": "primary/puf_tax_detail",
        "stage": "primary_puf_qrf",
        "entities": ["person", "tax_unit"],
        "donor": {"support_role": "puf_tax_detail"},
        "recipient_contract": {"scope": "whole_pool"},
        "model": "regime_gated_qrf",
        "predictors": ["puf_tax_detail"],
        "chaining": "base_plus_preceding_declared_targets",
        "execution_contract": "primary_puf_qrf",
        "targets": target_rows,
    }


def _late_families(
    metadata: PolicyEngineUSVariableMetadataIndex,
    *,
    resource_semantics: Mapping[str, object],
) -> list[dict[str, object]]:
    semantics_by_producer = {
        row["producer"]: row for row in resource_semantics["producers"]
    }
    families: list[dict[str, object]] = []
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        family_id = f"late/{group.entity}/{group.family}"
        model_resource = semantics_by_producer[group.name]["resources"][
            f"{group.entity}.@late_transfer_model_config"
        ]
        runtime_binding = model_resource["binding"]
        families.append(
            {
                "id": family_id,
                "stage": "late_producer_dag",
                "entities": [group.entity],
                "runtime_name": group.name,
                "donor": {
                    "channel": runtime_binding["donor_projection"]["support_channel"]
                },
                "donor_contract": {
                    "spine": runtime_binding["donor_spine"],
                    "resource_binding_channel": runtime_binding["donor_channel"],
                    "selection": runtime_binding["donor_selection"],
                    "projection": runtime_binding["donor_projection"],
                },
                "recipient_contract": {
                    "selection": (
                        "target_specific_complement_of_declared_producer_rows"
                    )
                },
                "model": "regime_gated_qrf",
                "predictors": _acs_predictor_blocks(entity=group.entity),
                "max_targets_per_fit": runtime_binding["max_targets_per_fit"],
                "execution_contract": "acs_transfer_default",
                "targets": [
                    {
                        **_target(metadata, entity=group.entity, name=target),
                        "output_coverage_scope": _canonical_output_coverage_scope(
                            producer=group.name,
                            entity=group.entity,
                            column=target,
                        ),
                    }
                    for target in group.targets
                ],
            }
        )
    return families


def _normalise_transfer_execution(
    base_contract: Mapping[str, object],
) -> dict[str, object]:
    """Factor shared ACS-transfer ABI fields into one typed declaration.

    Predictor arrays are already declared by the named predictor blocks, and
    the two post-transfer structures differ only by target-triggered feature
    activation.  Keeping those facts once avoids embedding the same full
    runtime identity in every late family and virtual resource.
    """

    result = _without_sha256(base_contract)
    result.pop("person_required_predictors")
    result.pop("person_optional_predictors")
    result.pop("group_required_predictors")
    housing = _mapping_like(result["housing"], "transfer execution housing")
    housing = deepcopy(dict(housing))
    housing.pop("mandatory_features")
    result["housing"] = housing
    post_transfer = _mapping_like(
        result.pop("post_transfer_structure"),
        "transfer execution post-transfer structure",
    )

    early_targets = _ordered_direction_targets(stacked_gap_fill_plan()[0])
    enabled_schedule = acs_transfer_execution_contract_identity(
        targets=early_targets,
        derive_schedule_d=True,
    )["post_transfer_structure"]["schedule_d_capital_gain_distributions"]
    schedule_base = deepcopy(
        dict(
            _mapping_like(
                post_transfer["schedule_d_capital_gain_distributions"],
                "schedule-D base contract",
            )
        )
    )
    schedule_base.pop("enabled")
    schedule_override = deepcopy(
        dict(
            _mapping_like(
                enabled_schedule,
                "schedule-D enabled contract",
            )
        )
    )
    schedule_override.pop("enabled")
    for key, value in list(schedule_override.items()):
        if schedule_base.get(key) == value:
            schedule_override.pop(key)

    adult_care = deepcopy(
        dict(
            _mapping_like(
                post_transfer["adult_care"],
                "adult-care post-transfer contract",
            )
        )
    )
    adult_care.pop("enabled")
    result["predictor_bindings"] = {
        "person_required": "acs_person_required",
        "person_optional": [
            "acs_person_optional_income",
            "acs_person_optional_structure",
        ],
        "group_required": "acs_group_required",
        "housing_mandatory": "acs_person_housing_required",
    }
    result["post_transfer_features"] = {
        "adult_care": {
            "activation": {
                "all_targets": [
                    "is_incapable_of_self_care",
                    "pre_subsidy_care_expenses",
                ]
            },
            "contract": adult_care,
        },
        "schedule_d_capital_gain_distributions": {
            "activation": {
                "derive_schedule_d": True,
                "all_targets": ["long_term_capital_gains_before_response"],
            },
            "contract": schedule_base,
            "enabled_overrides": schedule_override,
        },
    }
    result["profiles"] = {
        "acs_transfer_default": {"derive_schedule_d": False},
        "acs_transfer_early": {"derive_schedule_d": True},
    }
    return result


def _project_transfer_execution_identity(
    transfer_execution: Mapping[str, object],
    predictor_blocks: Mapping[str, object],
    *,
    profile_id: str,
    targets: Sequence[str],
) -> dict[str, object]:
    """Rehydrate one exact constants-era ACS-transfer ABI identity."""

    result = {
        key: deepcopy(value)
        for key, value in transfer_execution.items()
        if key not in {"predictor_bindings", "post_transfer_features", "profiles"}
    }
    bindings = _mapping_like(
        transfer_execution["predictor_bindings"], "transfer predictor bindings"
    )

    def block_columns(block_id: object) -> list[object]:
        block = _mapping_like(
            predictor_blocks[str(block_id)], f"predictor block {block_id}"
        )
        return deepcopy(list(_array_like(block["columns"], f"{block_id} columns")))

    result["person_required_predictors"] = block_columns(bindings["person_required"])
    result["person_optional_predictors"] = [
        column
        for block_id in _array_like(
            bindings["person_optional"], "person optional predictor block refs"
        )
        for column in block_columns(block_id)
    ]
    result["group_required_predictors"] = block_columns(bindings["group_required"])
    housing = deepcopy(dict(_mapping_like(result["housing"], "transfer housing")))
    housing["mandatory_features"] = block_columns(bindings["housing_mandatory"])
    result["housing"] = housing

    profiles = _mapping_like(transfer_execution["profiles"], "transfer profiles")
    profile = _mapping_like(profiles[profile_id], f"transfer profile {profile_id}")
    derive_schedule_d = bool(profile["derive_schedule_d"])
    target_set = set(targets)
    post_transfer: dict[str, object] = {}
    features = _mapping_like(
        transfer_execution["post_transfer_features"], "post-transfer features"
    )
    for feature_id, feature_value in features.items():
        feature = _mapping_like(feature_value, f"post-transfer feature {feature_id}")
        activation = _mapping_like(
            feature["activation"], f"post-transfer feature {feature_id} activation"
        )
        enabled = set(activation.get("all_targets", [])).issubset(target_set)
        if activation.get("derive_schedule_d") is True:
            enabled = enabled and derive_schedule_d
        contract = deepcopy(
            dict(
                _mapping_like(
                    feature["contract"], f"post-transfer feature {feature_id} contract"
                )
            )
        )
        if enabled:
            contract.update(deepcopy(dict(feature.get("enabled_overrides", {}))))
        contract["enabled"] = enabled
        post_transfer[feature_id] = contract
    result["post_transfer_structure"] = post_transfer
    result["sha256"] = _canonical_sha256(result)
    return result


def _worker_execution_template() -> dict[str, object]:
    """Describe exact runtime resolution with a closed resolver-op algebra."""

    return {
        "surface": "execution_profile",
        "resolve_as": "worker_execution",
        "template": {
            "module": "microcosm.build.us_runtime.puf_qrf_worker",
            "argv_template": [
                {"resolver_op": "sys_executable"},
                "-m",
                "microcosm.build.us_runtime.puf_qrf_worker",
                "--checkpoint-dir",
                "{checkpoint_dir}",
                "--target-index",
                "{target_index}",
            ],
            "interpreter": {
                "executable": {"resolver_op": "sys_executable"},
                "resolved_executable": {"resolver_op": "resolved_sys_executable"},
                "implementation": {"resolver_op": "python_implementation"},
                "cache_tag": {"resolver_op": "python_cache_tag"},
                "version": {"resolver_op": "python_version_triplet"},
            },
            "environment": {
                "policy": "inherit_parent_environment_with_bound_fit_controls",
                "overrides": {},
                "semantic_controls": {
                    "POPULACE_FIT_N_JOBS": {
                        "configured": {
                            "resolver_op": "environment_value",
                            "name": "POPULACE_FIT_N_JOBS",
                        },
                        "resolved": {
                            "resolver_op": "env_canonical_positive_int_or_default",
                            "name": "POPULACE_FIT_N_JOBS",
                            "default": -1,
                        },
                    },
                    "POPULACE_FIT_PREDICT_WORKERS": {
                        "configured": {
                            "resolver_op": "environment_value",
                            "name": "POPULACE_FIT_PREDICT_WORKERS",
                        },
                        "resolved": {
                            "resolver_op": "env_positive_int_or_cpu_count",
                            "name": "POPULACE_FIT_PREDICT_WORKERS",
                            "fallback_minimum": 1,
                        },
                        "resolution": {
                            "resolver_op": "env_or_cpu_count_resolution_label",
                            "name": "POPULACE_FIT_PREDICT_WORKERS",
                        },
                    },
                },
                "bound_names": [
                    "POPULACE_FIT_N_JOBS",
                    "POPULACE_FIT_PREDICT_WORKERS",
                ],
            },
        },
    }


def _resolve_worker_execution(value: Mapping[str, object]) -> dict[str, object]:
    """Resolve the one reviewed worker template, refusing an extended mini-language."""

    expected = _worker_execution_template()
    if value != expected:
        raise RuntimeError(
            "Primary-QRF worker template differs from the closed reviewed resolver."
        )
    fit_jobs_raw = os.environ.get("POPULACE_FIT_N_JOBS")
    if fit_jobs_raw is None:
        fit_jobs = -1
    else:
        try:
            fit_jobs = int(fit_jobs_raw)
        except ValueError as error:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from error
        if fit_jobs < 1 or str(fit_jobs) != fit_jobs_raw:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a canonical positive integer for "
                "the primary-QRF worker binding."
            )
    predict_workers_raw = os.environ.get("POPULACE_FIT_PREDICT_WORKERS")
    if predict_workers_raw is None or not predict_workers_raw.strip():
        predict_workers = os.cpu_count() or 1
        predict_workers_source = "os_cpu_count_fallback"
    else:
        try:
            predict_workers = int(predict_workers_raw)
        except ValueError as error:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from error
        if predict_workers < 1:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be positive for the "
                "primary-QRF worker binding."
            )
        predict_workers_source = "environment_override"
    executable = Path(sys.executable)
    module = "microcosm.build.us_runtime.puf_qrf_worker"
    return {
        "module": module,
        "argv_template": [
            str(executable),
            "-m",
            module,
            "--checkpoint-dir",
            "{checkpoint_dir}",
            "--target-index",
            "{target_index}",
        ],
        "interpreter": {
            "executable": str(executable),
            "resolved_executable": str(executable.resolve()),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:3]),
        },
        "environment": {
            "policy": "inherit_parent_environment_with_bound_fit_controls",
            "overrides": {},
            "semantic_controls": {
                "POPULACE_FIT_N_JOBS": {
                    "configured": fit_jobs_raw,
                    "resolved": fit_jobs,
                },
                "POPULACE_FIT_PREDICT_WORKERS": {
                    "configured": predict_workers_raw,
                    "resolved": predict_workers,
                    "resolution": predict_workers_source,
                },
            },
            "bound_names": [
                "POPULACE_FIT_N_JOBS",
                "POPULACE_FIT_PREDICT_WORKERS",
            ],
        },
    }


def _portable_resource_semantics(*, n_estimators: int) -> dict[str, object]:
    """Return late resource semantics with machine facts parameterized."""

    resolved = _json_ready(
        stacked_late_producer_resource_semantics_receipt(
            clone_attachment_fraction=_LEGACY_CLONE_ATTACHMENT_FRACTION,
            clone_attachment_seed=_LEGACY_CLONE_ATTACHMENT_SEED,
            primary_seed=POOL_RANDOM_SEED,
            primary_n_estimators=n_estimators,
            transfer_seed=POOL_RANDOM_SEED,
            transfer_n_estimators=n_estimators,
            transfer_max_targets_per_fit=(DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT),
        )
    )
    resolved.pop("sha256")
    primary = next(
        row for row in resolved["producers"] if row["producer"] == "primary_puf_qrf"
    )
    primary_execution = primary["resources"]["tax_unit.@primary_puf_execution_config"][
        "binding"
    ]
    primary_execution["qrf"]["worker_execution"] = _worker_execution_template()
    source_bindings = [
        resource["binding"]
        for producer in resolved["producers"]
        for resource in producer["resources"].values()
        if resource["binding"]["resource_kind"] == "post_clone_source_execution_config"
    ]
    registries = {tuple(binding["operator_registry"]) for binding in source_bindings}
    phases = {binding["phase"] for binding in source_bindings}
    seeds = {binding["seed"] for binding in source_bindings}
    periods = {
        binding["time_period"]
        for binding in source_bindings
        if binding["time_period"] is not None
    }
    stage_specs = [
        binding["source_stage_spec"]
        for binding in source_bindings
        if binding["source_stage_spec"] is not None
    ]
    asset_rows = {
        (
            stage["asset"],
            stage["asset_sha256"],
            json.dumps(stage["manifest"], sort_keys=True, separators=(",", ":")),
        )
        for stage in stage_specs
    }
    if len(registries) != 1 or len(phases) != 1 or len(seeds) != 1 or len(periods) != 1:
        raise RuntimeError("Late-source shared execution defaults are inconsistent.")
    if len(asset_rows) != 1:
        raise RuntimeError(
            "Late-source stage bindings do not share one asset identity."
        )
    if phases != {"post_clone"} or seeds != {POOL_RANDOM_SEED}:
        raise RuntimeError("Late-source phase or seed changed from the typed refs.")
    resolved["source_execution_defaults"] = {
        "operator_registry_ref": deepcopy(_SOURCE_OPERATOR_REGISTRY_REF),
        "phase": next(iter(phases)),
        "time_period_ref": deepcopy(_TARGET_PERIOD_REF),
    }
    resolved["resolution"] = {
        "clone_attachment_ref": deepcopy(_PUF_ATTACHMENT_REF),
        "build_model_seed_ref": deepcopy(_BUILD_MODEL_SEED_REF),
        "digest_rule": "canonical_sha256(resolved_payload_without_sha256)",
    }
    return resolved


def _node_kernel(name: str, kind: str) -> str:
    if kind == "acs_earnings_universe":
        return "kernel:acs_pums_earnings_universe"
    if kind == "primary_puf":
        return "kernel:primary_puf_qrf"
    if kind == "post_clone_source":
        if not name.startswith("source:"):
            raise RuntimeError(f"Unexpected source producer name {name!r}.")
        return f"kernel:{name.removeprefix('source:')}"
    if kind == "source_finalizer":
        return "kernel:source_finalizer"
    if kind == "late_transfer":
        return "kernel:acs_transfer"
    raise RuntimeError(f"Unexpected US late producer kind {kind!r}.")


_VIRTUAL_RESOURCE_SURFACE_KINDS = {
    "acs_pums_earnings_universe_execution_config": "execution_config",
    "late_transfer_model_config": "execution_config",
    "late_transfer_target_bank": "target_bank",
    "post_clone_source_execution_config": "execution_config",
    "primary_puf_execution_config": "execution_config",
    "primary_qrf_checkpoint": "target_bank",
    "puf_donor_tax_units": "manifest",
    "source_finalizer_execution_config": "execution_config",
    "source_operator_receipt": "producer_receipt",
}


def _normalise_virtual_resource(
    *,
    producer: str,
    resource_id: str,
    resource: Mapping[str, object],
) -> dict[str, object]:
    """Turn a runtime resource-map entry into the approved typed resource row."""

    result = deepcopy(dict(resource))
    binding = deepcopy(
        dict(_mapping_like(result["binding"], f"resource {resource_id} binding"))
    )
    resource_kind = str(binding["resource_kind"])
    try:
        surface_kind = _VIRTUAL_RESOURCE_SURFACE_KINDS[resource_kind]
    except KeyError as error:
        raise RuntimeError(
            f"US resource {producer}/{resource_id} has unknown kind {resource_kind!r}."
        ) from error

    if resource_kind == "late_transfer_model_config":
        expected_producer = binding.pop("producer")
        if expected_producer != producer:
            raise RuntimeError(
                f"Late-transfer resource producer {expected_producer!r} does not "
                f"match graph producer {producer!r}."
            )
        for duplicate in (
            "n_estimators",
            "entity",
            "family",
            "ordered_targets",
            "max_targets_per_fit",
            "donor_spine",
            "donor_channel",
            "donor_selection",
            "donor_projection",
            "transfer_execution_contract",
        ):
            binding.pop(duplicate)
        if binding.pop("seed") != POOL_RANDOM_SEED:
            raise RuntimeError("Late-transfer seed differs from build_model_seed.")
    elif resource_kind == "primary_puf_execution_config":
        clone_attachment = _mapping_like(
            binding.pop("clone_attachment"), "primary clone attachment"
        )
        if (
            clone_attachment.get("fraction") != _LEGACY_CLONE_ATTACHMENT_FRACTION
            or clone_attachment.get("seed") != _LEGACY_CLONE_ATTACHMENT_SEED
            or clone_attachment.get("puf_clone_index") != 1
            or clone_attachment.get("support_channels")
            != ["asec", "puf_tax_detail"]
        ):
            raise RuntimeError(
                "Primary clone attachment differs from the spine-owned contract."
            )
        qrf = deepcopy(dict(_mapping_like(binding["qrf"], "primary QRF binding")))
        for duplicate in (
            "n_estimators",
            "predictors",
            "person_outputs",
            "tax_unit_outputs",
        ):
            qrf.pop(duplicate)
        if qrf.pop("seed") != POOL_RANDOM_SEED:
            raise RuntimeError("Primary QRF seed differs from build_model_seed.")
        tail = deepcopy(
            dict(_mapping_like(binding["capital_gains_tail"], "capital-gains tail"))
        )
        if tail.pop("seed") != POOL_RANDOM_SEED:
            raise RuntimeError("Capital-gains tail seed differs from build_model_seed.")
        tail.pop("support_contract")
        tail["support_contract_ref"] = deepcopy(_PUF_TAIL_SUPPORT_REF)
        soi_bands = deepcopy(
            dict(
                _mapping_like(
                    tail["soi_e19200_agi_bands"],
                    "capital-gains-tail SOI AGI bands",
                )
            )
        )
        runtime_bands = deepcopy(
            dict(
                _mapping_like(
                    soi_bands["runtime_agi_bands"],
                    "capital-gains-tail runtime AGI bands",
                )
            )
        )
        runtime_bands.pop("sha256")
        soi_bands["runtime_agi_bands"] = runtime_bands
        tail["soi_e19200_agi_bands"] = soi_bands
        binding["qrf"] = qrf
        binding["capital_gains_tail"] = tail
    elif resource_kind == "post_clone_source_execution_config":
        operator = str(binding.pop("operator"))
        if producer != f"source:{operator}":
            raise RuntimeError(
                f"Source resource operator {operator!r} does not match {producer!r}."
            )
        for duplicate in (
            "declared_output_family",
            "operator_registry",
            "phase",
            "seed",
            "time_period",
        ):
            binding.pop(duplicate)
        stage_spec = binding.pop("source_stage_spec")
        if stage_spec is None:
            binding["source_stage_ref"] = None
        else:
            stage = _mapping_like(stage_spec, f"source stage binding for {producer}")
            resolver = _mapping_like(
                stage["runtime_stage_spec_resolver"],
                f"source stage resolver for {producer}",
            )
            binding["source_stage_ref"] = {
                "asset_id": "source_stages",
                "stage_id": stage["stage_name"],
                "runtime_resolver": {
                    "module": resolver["module"],
                    "callable": resolver["callable"],
                    "purpose": "legacy_runtime_verification_evidence",
                },
            }
    elif resource_kind == "primary_qrf_checkpoint":
        for duplicate in (
            "checkpoint_schema_version",
            "target_order",
            "target_order_sha256",
        ):
            binding.pop(duplicate)
    elif resource_kind == "late_transfer_target_bank":
        dynamic_field = deepcopy(
            dict(
                _mapping_like(
                    result["dynamic_field"],
                    f"late-transfer target-bank dynamic field for {producer}",
                )
            )
        )
        derivation = deepcopy(
            dict(
                _mapping_like(
                    dynamic_field["derivation"],
                    f"late-transfer target-bank derivation for {producer}",
                )
            )
        )
        derivation.pop("late_producer_dag_sha256")
        derivation.pop("late_producer_schedule_sha256")
        dynamic_field["derivation"] = derivation
        result["dynamic_field"] = dynamic_field
    elif resource_kind == "acs_pums_earnings_universe_execution_config":
        identity = deepcopy(
            dict(
                _mapping_like(
                    binding["contract_identity"],
                    "ACS PUMS earnings-universe contract identity",
                )
            )
        )
        identity.pop("sha256")
        binding["contract_identity"] = identity

    result["binding"] = binding
    return {
        "id": resource_id,
        "kind": surface_kind,
        **result,
    }


_NO_WRITE_ACTIONS = {
    "consume_only_byte_exact_noop",
    "origin_projection_masked_noop",
    "producer_masked_byte_exact_noop",
    "scope_masked_noop",
}


def _typed_output(output: Mapping[str, object]) -> dict[str, object]:
    """Add planner lifetime flags without changing the legacy contract payload."""

    result = deepcopy(dict(output))
    column = str(output["column"])
    is_ephemeral_receipt = column.startswith("@source_receipt:") or column == (
        "@acs_pums_earnings_universe_application"
    )
    result["temporary"] = is_ephemeral_receipt
    result["validation_only"] = column.startswith("@") and column != "@resolved_weight"
    return result


def _output_key(output: Mapping[str, object]) -> tuple[str, str]:
    return str(output["entity"]), str(output["column"])


def _family_producer_name(family: Mapping[str, object]) -> str | None:
    stage = str(family["stage"])
    if stage == "primary_puf_qrf":
        return str(family["execution_contract"])
    if stage == "late_producer_dag":
        return str(family["runtime_name"])
    return None


def _family_owned_outputs_by_producer(
    families: Sequence[object],
) -> dict[str, list[dict[str, object]]]:
    """Compile modeled node outputs from their single family authority."""

    result: dict[str, list[dict[str, object]]] = {}
    family_ids: dict[str, str] = {}
    for value in families:
        family = _mapping_like(value, "imputation family")
        producer = _family_producer_name(family)
        if producer is None:
            for target_value in _array_like(family["targets"], "family targets"):
                target = _mapping_like(target_value, "family target")
                if "output_coverage_scope" in target:
                    raise RuntimeError(
                        "Only primary and late families may declare producer output "
                        f"coverage: {family['id']!r}."
                    )
            continue
        if producer in result:
            raise RuntimeError(
                "Producer node is linked from more than one imputation family: "
                f"{producer!r} by {family_ids[producer]!r} and {family['id']!r}."
            )
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for target_value in _array_like(family["targets"], "family targets"):
            target = _mapping_like(target_value, "family target")
            if "output_coverage_scope" not in target:
                raise RuntimeError(
                    f"Family {family['id']!r} target {target['name']!r} is missing "
                    "output_coverage_scope."
                )
            # Family target names use the closed identifier grammar, so they
            # cannot name virtual ``@`` receipts.  Modeled outputs are therefore
            # always persistent materialized columns by schema, not convention.
            output = {
                "entity": target["entity"],
                "column": target["name"],
                "coverage_scope": target["output_coverage_scope"],
                "temporary": False,
                "validation_only": False,
            }
            key = _output_key(output)
            if key in seen:
                raise RuntimeError(
                    f"Family {family['id']!r} repeats producer output {key!r}."
                )
            seen.add(key)
            rows.append(output)
        result[producer] = rows
        family_ids[producer] = str(family["id"])
    return result


def _assert_family_node_kinds(
    *,
    nodes_by_name: Mapping[str, Mapping[str, object]],
    family_outputs: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    for producer in family_outputs:
        if producer not in nodes_by_name:
            raise RuntimeError(
                f"Imputation family references missing producer node {producer!r}."
            )
        node = nodes_by_name[producer]
        expected_kind = (
            "primary_puf" if producer == "primary_puf_qrf" else "late_transfer"
        )
        if node["kind"] != expected_kind:
            raise RuntimeError(
                f"Family-linked producer {producer!r} must have kind "
                f"{expected_kind!r}, got {node['kind']!r}."
            )
    unlinked = sorted(
        name
        for name, node in nodes_by_name.items()
        if node["kind"] in {"primary_puf", "late_transfer"}
        and name not in family_outputs
    )
    if unlinked:
        raise RuntimeError(
            "Modeled producer nodes must resolve exactly one family: "
            f"{unlinked!r}."
        )


def _strip_family_owned_node_outputs(
    *,
    nodes: Sequence[Mapping[str, object]],
    families: Sequence[object],
) -> list[dict[str, object]]:
    """Remove constants-era family mirrors from generated authored nodes."""

    family_outputs = _family_owned_outputs_by_producer(families)
    result = [deepcopy(dict(node)) for node in nodes]
    nodes_by_name = {str(node["name"]): node for node in result}
    if len(nodes_by_name) != len(result):
        raise RuntimeError("Producer graph repeats a node name.")
    _assert_family_node_kinds(
        nodes_by_name=nodes_by_name,
        family_outputs=family_outputs,
    )
    for producer, expected_rows in family_outputs.items():
        node = nodes_by_name[producer]
        authored_rows = [
            _mapping_like(value, f"{producer} output")
            for value in _array_like(node["outputs"], f"{producer} outputs")
        ]
        by_key: dict[tuple[str, str], Mapping[str, object]] = {}
        for row in authored_rows:
            key = _output_key(row)
            if key in by_key:
                raise RuntimeError(
                    f"Canonical producer {producer!r} repeats output {key!r}."
                )
            by_key[key] = row
        expected_by_key = {_output_key(row): row for row in expected_rows}
        for key, expected in expected_by_key.items():
            actual = by_key.get(key)
            if actual is None:
                raise RuntimeError(
                    f"Family-owned output {producer}/{key!r} is absent from the "
                    "constants-era graph."
                )
            if dict(actual) != dict(expected):
                raise RuntimeError(
                    f"Family-owned output {producer}/{key!r} conflicts with the "
                    f"constants-era graph: family={dict(expected)!r}, "
                    f"graph={dict(actual)!r}."
                )
        node["outputs"] = [
            deepcopy(dict(row))
            for row in authored_rows
            if _output_key(row) not in expected_by_key
        ]
    return result


def _compile_node_outputs(
    document: Mapping[str, object],
    graph: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Expand family-owned outputs and restore canonical node output order."""

    if graph is not document.get("producer_graph"):
        raise RuntimeError("Producer graph must be the document's authored graph.")
    return {
        producer: [deepcopy(dict(row)) for row in rows]
        for producer, rows in compile_producer_outputs(
            {"imputation": document}
        ).items()
    }


def _node_capabilities(kind: str) -> dict[str, object]:
    if kind == "acs_earnings_universe":
        return {
            "determinism": "deterministic",
            "numeric_reproducibility": "bitwise",
            "effects": ["declared_source_read"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    if kind == "primary_puf":
        return {
            "determinism": "seeded",
            "numeric_reproducibility": "tolerance_bound",
            "effects": ["declared_source_read", "declared_sink_write"],
            "structural_delta": "expand",
            "retry_safety": "attempt_scoped",
        }
    if kind == "post_clone_source":
        return {
            "determinism": "seeded",
            "numeric_reproducibility": "tolerance_bound",
            "effects": ["declared_source_read"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    if kind == "source_finalizer":
        return {
            "determinism": "deterministic",
            "numeric_reproducibility": "bitwise",
            "effects": ["none"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    if kind == "late_transfer":
        return {
            "determinism": "seeded",
            "numeric_reproducibility": "tolerance_bound",
            "effects": ["none"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    raise RuntimeError(f"Unexpected producer kind {kind!r}.")


def _node_mutations(kind: str) -> dict[str, object]:
    """Declare the structural Frame contract observed at the current seam."""

    if kind != "primary_puf":
        return {
            "entity_keys": {
                "operation": "preserve",
                "precondition": "entity_keys_valid",
                "postcondition": "entity_keys_unchanged",
            },
            "cardinality": {
                "operation": "preserve",
                "precondition": "entity_cardinality_valid",
                "postcondition": "entity_cardinality_unchanged",
            },
            "links": {
                "operation": "preserve",
                "precondition": "links_valid",
                "postcondition": "links_unchanged",
            },
            "memberships": {
                "operation": "preserve",
                "precondition": "memberships_valid",
                "postcondition": "memberships_unchanged",
            },
            "order": {
                "operation": "preserve",
                "precondition": "entity_order_valid",
                "postcondition": "entity_order_unchanged",
            },
            "weights": {
                "operation": "preserve",
                "precondition": "weights_valid",
                "postcondition": "weights_unchanged",
            },
            "mass_history": {
                "operation": "preserve",
                "precondition": "mass_history_valid",
                "postcondition": "mass_history_unchanged",
            },
        }
    return {
        "entity_keys": {
            "operation": "append_remapped_clone_keys",
            "precondition": "native_entity_keys_unique",
            "postcondition": "all_entity_keys_unique",
        },
        "cardinality": {
            "operation": "expand_complete_household_graphs",
            "precondition": "native_clone_index_zero",
            "postcondition": "clone_roles_materialized",
        },
        "links": {
            "operation": "preserve_absent",
            "precondition": "link_tables_absent",
            "postcondition": "link_tables_absent",
        },
        "memberships": {
            "operation": "append_relinked_clone_memberships",
            "precondition": "native_memberships_valid",
            "postcondition": "clone_memberships_reference_remapped_keys",
        },
        "order": {
            "operation": "append_clone_blocks_preserving_native_order",
            "precondition": "native_entity_order_valid",
            "postcondition": "clone_blocks_follow_native_rows",
        },
        "weights": {
            "operation": "split_mass_across_clone_descendants",
            "precondition": "native_household_mass_finite",
            "postcondition": "household_mass_conserved",
        },
        "mass_history": {
            "operation": "preserve",
            "precondition": "mass_history_valid",
            "postcondition": "mass_history_unchanged",
        },
    }


def _write_scope(
    *,
    producer: str,
    output: Mapping[str, object],
    ownership_rows: Sequence[object],
) -> dict[str, object]:
    entity = str(output["entity"])
    column = str(output["column"])
    segments = []
    matching_rows = [
        _mapping_like(value, "overlap ownership row")
        for value in ownership_rows
        if _mapping_like(value, "overlap ownership row")["entity"] == entity
        and _mapping_like(value, "overlap ownership row")["target"] == column
    ]
    for row in matching_rows:
        actions = [
            _mapping_like(value, "overlap producer action")
            for value in _array_like(row["producer_actions"], "producer actions")
            if _mapping_like(value, "overlap producer action")["producer"] == producer
        ]
        if len(actions) != 1:
            raise RuntimeError(
                f"Ownership row {entity}.{column} does not name {producer!r} once."
            )
        action = str(actions[0]["action"])
        if action not in _NO_WRITE_ACTIONS:
            segments.append(
                {
                    "predicate": "origin_clone",
                    "origin": row["origin"],
                    "clone_index": row["clone_index"],
                    "write_policy": action,
                }
            )
    if not matching_rows:
        segments.append(
            {
                "predicate": "coverage_scope",
                "coverage_scope": output["coverage_scope"],
                "write_policy": "declared_output_write",
            }
        )
    if not segments:
        raise RuntimeError(
            f"Producer {producer!r} declares {entity}.{column} but owns no cells."
        )
    if column == "@resolved_weight":
        mode = "resolved_weight"
    elif column.startswith("@"):
        mode = "virtual_receipt"
    elif column.endswith("_id") or "support_" in column:
        mode = "structural_column"
    else:
        mode = "column_cells"
    return {
        "entity": entity,
        "column": column,
        "row_scope": output["coverage_scope"],
        "mode": mode,
        "cell_segments": segments,
    }


def _segments_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    scope_coverage: Mapping[str, object],
) -> bool:
    if left["predicate"] == right["predicate"] == "origin_clone":
        return (left["origin"], left["clone_index"]) == (
            right["origin"],
            right["clone_index"],
        )
    declared = _mapping_like(scope_coverage["declared"], "scope coverage")

    def atoms(segment: Mapping[str, object]) -> set[tuple[str, int]]:
        if segment["predicate"] == "origin_clone":
            return {(str(segment["origin"]), int(segment["clone_index"]))}
        scope = str(segment["coverage_scope"])
        scopes = set(_array_like(declared.get(scope, [scope]), f"scope {scope}"))
        result: set[tuple[str, int]] = set()
        if "whole_pool" in scopes:
            return {
                (origin, clone_index)
                for origin in ("asec", "acs")
                for clone_index in (0, 1, 2)
            }
        if "asec_source" in scopes:
            result.add(("asec", 0))
        if "acs_source" in scopes:
            result.add(("acs", 0))
        if "puf_clone" in scopes:
            result.update(
                (origin, clone_index)
                for origin in ("asec", "acs")
                for clone_index in (1, 2)
            )
        if "receipt" in scopes:
            result.add(("receipt", 0))
        return result

    return bool(atoms(left) & atoms(right))


def _incomparable_proof(
    *,
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[object],
    scope_coverage: Mapping[str, object],
) -> dict[str, object]:
    names = [str(node["name"]) for node in nodes]
    reachable = {name: set() for name in names}
    for value in edges:
        edge = _array_like(value, "producer graph edge")
        reachable[str(edge[0])].add(str(edge[1]))
    changed = True
    while changed:
        changed = False
        for name in names:
            expanded = set(reachable[name])
            for child in tuple(reachable[name]):
                expanded.update(reachable[child])
            if expanded != reachable[name]:
                reachable[name] = expanded
                changed = True
    incomparable = 0
    disjoint = 0
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            left_name = str(left["name"])
            right_name = str(right["name"])
            if right_name in reachable[left_name] or left_name in reachable[right_name]:
                continue
            incomparable += 1
            overlap = False
            for left_scope_value in _array_like(
                left["write_scopes"], f"{left_name} write scopes"
            ):
                left_scope = _mapping_like(left_scope_value, "left write scope")
                for right_scope_value in _array_like(
                    right["write_scopes"], f"{right_name} write scopes"
                ):
                    right_scope = _mapping_like(right_scope_value, "right write scope")
                    if (left_scope["entity"], left_scope["column"]) != (
                        right_scope["entity"],
                        right_scope["column"],
                    ):
                        continue
                    if any(
                        _segments_overlap(
                            _mapping_like(a, "left cell segment"),
                            _mapping_like(b, "right cell segment"),
                            scope_coverage=scope_coverage,
                        )
                        for a in _array_like(
                            left_scope["cell_segments"], "left cell segments"
                        )
                        for b in _array_like(
                            right_scope["cell_segments"], "right cell segments"
                        )
                    ):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                raise RuntimeError(
                    "Incomparable producer nodes have overlapping exact writes: "
                    f"{left_name!r}, {right_name!r}."
                )
            disjoint += 1
    return {
        "requirement": "commute_or_disjoint_writes",
        "proof_method": "transitive_closure_and_closed_cell_segment_intersection",
        "overlap_rule": "explicit_commutativity_proof_required",
        "commutativity_proofs": [],
        "incomparable_pair_count": incomparable,
        "disjoint_write_pair_count": disjoint,
    }


def _producer_graph(
    *,
    schedule_receipt: Mapping[str, object],
    resource_semantics: Mapping[str, object],
    ownership: Mapping[str, object],
    families: Sequence[object],
) -> dict[str, object]:
    canonical = json.loads(CANONICAL_US_LATE_PRODUCER_SCHEDULE.canonical_json)
    if _canonical_sha256(canonical) != CANONICAL_US_LATE_PRODUCER_SCHEDULE.sha256:
        raise RuntimeError("Canonical US late producer graph digest changed in memory.")
    resources_by_producer = {
        row["producer"]: row["resources"] for row in resource_semantics["producers"]
    }
    nodes = []
    for contract in canonical["contracts"]:
        name = contract["name"]
        nodes.append(
            {
                "id": name,
                "name": name,
                "kind": contract["kind"],
                "kernel": _node_kernel(name, contract["kind"]),
                "capabilities": _node_capabilities(contract["kind"]),
                "mutations": _node_mutations(contract["kind"]),
                "inputs": contract["inputs"],
                "outputs": [_typed_output(output) for output in contract["outputs"]],
                "virtual_resources": [
                    _normalise_virtual_resource(
                        producer=name,
                        resource_id=resource_id,
                        resource=_mapping_like(
                            resource,
                            f"resource semantics {name}/{resource_id}",
                        ),
                    )
                    for resource_id, resource in resources_by_producer[name].items()
                ],
            }
        )
    nodes = _strip_family_owned_node_outputs(nodes=nodes, families=families)
    overlap_contract = {
        key: deepcopy(value)
        for key, value in ownership.items()
        if key not in {"ownership", "sha256", "targets"}
    }
    return {
        "graph_schema_version": canonical["schema_version"],
        "schedule_payload_schema_version": schedule_receipt["schema_version"],
        "external_stages": canonical["external_stages"],
        "scope_coverage": canonical["scope_coverage"],
        "scope_registry": producer_scope_registry_wire(),
        "nodes": nodes,
        "execution_receipt_contract": schedule_receipt["execution_receipt_contract"],
        "ownership_contract": overlap_contract,
        "ownership_matrix": ownership["ownership"],
        "resource_semantics": {
            "artifact_kind": resource_semantics["artifact_kind"],
            "schema_version": resource_semantics["schema_version"],
            "resolution": resource_semantics["resolution"],
            "source_execution_defaults": resource_semantics[
                "source_execution_defaults"
            ],
        },
    }


def _normalise_gap_fill_schedule(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Keep only non-derivable producer scheduling facts."""

    directions: list[dict[str, object]] = []
    for direction_value in _array_like(
        receipt["directions"], "gap-fill producer schedule directions"
    ):
        direction = _mapping_like(direction_value, "gap-fill schedule direction")
        directions.append(
            {
                "name": direction["name"],
                "activation_stage": direction["activation_stage"],
            }
        )
    return {
        "activation_policy": receipt["status"],
        "directions": directions,
    }


def _project_gap_fill_plan(
    document: Mapping[str, object],
) -> list[dict[str, object]]:
    families = [
        _mapping_like(value, "imputation family")
        for value in _array_like(document["families"], "imputation families")
        if _mapping_like(value, "imputation family").get("stage")
        == "gap_fill_stacked_spine"
    ]
    schedule = _mapping_like(document["gap_fill_schedule"], "gap-fill schedule")
    result: list[dict[str, object]] = []
    for direction_value in _array_like(
        schedule["directions"], "gap-fill schedule directions"
    ):
        direction = _mapping_like(direction_value, "gap-fill schedule direction")
        name = str(direction["name"])
        direction_families = [
            family for family in families if family.get("direction") == name
        ]
        if not direction_families:
            raise RuntimeError(f"Gap-fill direction {name!r} has no families.")
        donor_channels = {
            str(family["donor"]["channel"]) for family in direction_families
        }
        recipient_channels = {
            str(family["recipient"]["channel"]) for family in direction_families
        }
        if len(donor_channels) != 1 or len(recipient_channels) != 1:
            raise RuntimeError(
                f"Gap-fill direction {name!r} has inconsistent channels."
            )
        target_families: dict[str, dict[str, list[str]]] = {}
        absence_rules: list[object] = []
        seen_absence_rules: set[str] = set()
        for family in direction_families:
            _, family_direction, entity, family_name = str(family["id"]).split("/", 3)
            if family_direction != name:
                raise RuntimeError(
                    f"Gap-fill family {family['id']!r} direction disagrees."
                )
            target_families.setdefault(entity, {})[family_name] = [
                str(target["name"])
                for target in _array_like(family["targets"], "family targets")
            ]
            for target in _array_like(family["targets"], "family targets"):
                target_row = _mapping_like(target, "family target")
                for rule in target_row.get("recipient_absence_rules", []):
                    key = json.dumps(rule, sort_keys=True, separators=(",", ":"))
                    if key not in seen_absence_rules:
                        seen_absence_rules.add(key)
                        absence_rules.append(deepcopy(rule))
        result.append(
            {
                "name": name,
                "donor_channel": donor_channels.pop(),
                "recipient_channel": recipient_channels.pop(),
                "target_families": target_families,
                "recipient_absence_rules": absence_rules,
            }
        )
    return result


def _project_gap_fill_schedule(
    document: Mapping[str, object],
    gap_fill_plan: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    schedule = _mapping_like(document["gap_fill_schedule"], "gap-fill schedule")
    plans = {str(plan["name"]): plan for plan in gap_fill_plan}
    directions = []
    total_targets = 0
    for index, direction_value in enumerate(
        _array_like(schedule["directions"], "gap-fill schedule directions")
    ):
        direction = _mapping_like(direction_value, "gap-fill schedule direction")
        plan = plans[str(direction["name"])]
        targets = []
        direction_families = [
            _mapping_like(value, "gap-fill family")
            for value in _array_like(document["families"], "imputation families")
            if _mapping_like(value, "imputation family").get("stage")
            == "gap_fill_stacked_spine"
            and _mapping_like(value, "imputation family").get("direction")
            == direction["name"]
        ]
        for family in direction_families:
            _, family_direction, family_entity, family_name = str(
                family["id"]
            ).split("/", 3)
            if family_direction != direction["name"]:
                raise RuntimeError(
                    f"Gap-fill family {family['id']!r} direction disagrees."
                )
            for target_value in _array_like(family["targets"], "family targets"):
                target_row = _mapping_like(target_value, "family target")
                producer = _mapping_like(
                    target_row["producer_binding"], "gap-fill producer binding"
                )
                targets.append(
                    {
                        "entity": family_entity,
                        "family": family_name,
                        "column": target_row["name"],
                        "producer": producer["operator"],
                        "producer_order_index": producer["order_index"],
                        "execution_scope": producer["execution_scope"],
                        "produced_channel": plan["donor_channel"],
                        "producer_stage": producer["stage"],
                    }
                )
        total_targets += len(targets)
        directions.append(
            {
                "name": direction["name"],
                "order_index": index,
                "donor_channel": plan["donor_channel"],
                "activation_stage": direction["activation_stage"],
                "target_count": len(targets),
                "targets": targets,
            }
        )
    result: dict[str, object] = {
        "status": schedule["activation_policy"],
        "direction_count": len(directions),
        "target_count": total_targets,
        "directions": directions,
    }
    result["sha256"] = _canonical_sha256(result)
    return result


def _primary_family_from_document(
    document: Mapping[str, object],
) -> Mapping[str, object]:
    matches = [
        _mapping_like(value, "primary imputation family")
        for value in _array_like(document["families"], "imputation families")
        if _mapping_like(value, "imputation family").get("stage") == "primary_puf_qrf"
    ]
    if len(matches) != 1:
        raise RuntimeError("Imputation document must have exactly one primary family.")
    return matches[0]


def _require_typed_ref(
    value: object,
    expected: Mapping[str, str],
    *,
    location: str,
) -> None:
    reference = _mapping_like(value, location)
    if dict(reference) != dict(expected):
        raise RuntimeError(
            f"{location} must be the reviewed typed reference; "
            f"expected={dict(expected)!r}, actual={dict(reference)!r}."
        )


def _spine_support_role(
    spine_document: Mapping[str, object],
    *,
    role_id: str,
) -> Mapping[str, object]:
    matches = [
        _mapping_like(value, "spine support role")
        for value in _array_like(spine_document["support_roles"], "spine support roles")
        if _mapping_like(value, "spine support role").get("id") == role_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Spine must declare exactly one support role {role_id!r}."
        )
    return matches[0]


def _build_model_seed_default(value: object, *, location: str) -> int:
    _require_typed_ref(value, _BUILD_MODEL_SEED_REF, location=location)
    defaults = {
        site.default
        for site in LEGACY_V1_PROTOCOL.sites
        if site.value_source == _BUILD_MODEL_SEED_REF["value_source"]
        and site.default is not None
    }
    if defaults != {POOL_RANDOM_SEED}:
        raise RuntimeError(
            "legacy-v1 build_model_seed sites no longer share one default: "
            f"{sorted(defaults)!r}."
        )
    return defaults.pop()


def _target_period(
    bundle_document: Mapping[str, object],
    value: object,
    *,
    location: str,
) -> int:
    _require_typed_ref(value, _TARGET_PERIOD_REF, location=location)
    dataset_run = _mapping_like(bundle_document["dataset_run"], "bundle dataset_run")
    period = dataset_run.get("target_period")
    if isinstance(period, bool) or not isinstance(period, int):
        raise RuntimeError("bundle dataset_run target_period must be an integer.")
    return period


def _post_clone_operator_order(
    spine_document: Mapping[str, object],
    value: object,
    *,
    location: str,
) -> list[str]:
    _require_typed_ref(value, _SOURCE_OPERATOR_REGISTRY_REF, location=location)
    pipeline = _mapping_like(
        spine_document["pipeline_contract"], "spine pipeline contract"
    )
    return [
        str(operator)
        for operator in _array_like(
            pipeline["post_clone_source_operator_order"],
            "spine post-clone source operator order",
        )
    ]


def _inflate_virtual_resource(
    *,
    document: Mapping[str, object],
    node: Mapping[str, object],
    resource_row: Mapping[str, object],
    late_families: Mapping[str, Mapping[str, object]],
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
    schedule_sha256: str,
    schedule_payload_sha256: str,
) -> tuple[str, dict[str, object]]:
    resource = {
        key: deepcopy(value)
        for key, value in resource_row.items()
        if key not in {"id", "kind"}
    }
    resource_id = str(resource_row["id"])
    binding = deepcopy(
        dict(_mapping_like(resource["binding"], f"resource {resource_id} binding"))
    )
    resource_kind = str(binding["resource_kind"])
    predictor_blocks = _mapping_like(
        document["predictor_blocks"], "imputation predictor blocks"
    )
    transfer_execution = _mapping_like(
        document["transfer_execution"], "imputation transfer execution"
    )
    models = _mapping_like(document["models"], "imputation models")
    qrf_model = _mapping_like(models["regime_gated_qrf"], "regime-gated QRF model")
    qrf_params = _mapping_like(qrf_model["params"], "regime-gated QRF parameters")
    graph = _mapping_like(document["producer_graph"], "producer graph")
    resource_semantics = _mapping_like(
        graph["resource_semantics"], "producer resource semantics"
    )
    resolution = _mapping_like(
        resource_semantics["resolution"], "producer resource resolution"
    )
    if resolution.get("digest_rule") != (
        "canonical_sha256(resolved_payload_without_sha256)"
    ):
        raise RuntimeError("Producer resource digest rule is unsupported.")
    build_model_seed = _build_model_seed_default(
        resolution["build_model_seed_ref"],
        location="resource semantics build_model_seed_ref",
    )

    if resource_kind == "late_transfer_model_config":
        producer = str(node["name"])
        family = late_families[producer]
        targets = [
            str(target["name"])
            for target in _array_like(family["targets"], "late family targets")
        ]
        entity = str(_array_like(family["entities"], "late family entities")[0])
        donor_contract = _mapping_like(
            family["donor_contract"], "late family donor contract"
        )
        binding.update(
            {
                "producer": producer,
                "n_estimators": qrf_params["n_estimators"],
                "entity": entity,
                "family": producer.split("/", 1)[1],
                "ordered_targets": targets,
                "max_targets_per_fit": family["max_targets_per_fit"],
                "donor_spine": donor_contract["spine"],
                "donor_channel": donor_contract["resource_binding_channel"],
                "donor_selection": donor_contract["selection"],
                "donor_projection": donor_contract["projection"],
                "transfer_execution_contract": _project_transfer_execution_identity(
                    transfer_execution,
                    predictor_blocks,
                    profile_id=str(family["execution_contract"]),
                    targets=targets,
                ),
            }
        )
        binding["seed"] = build_model_seed
    elif resource_kind == "primary_puf_execution_config":
        family = _primary_family_from_document(document)
        targets = [
            _mapping_like(value, "primary target")
            for value in _array_like(family["targets"], "primary targets")
        ]
        qrf = deepcopy(dict(_mapping_like(binding["qrf"], "primary QRF binding")))
        predictor_ref = str(_array_like(family["predictors"], "primary predictors")[0])
        qrf.update(
            {
                "n_estimators": qrf_params["n_estimators"],
                "predictors": deepcopy(
                    list(
                        _array_like(
                            _mapping_like(
                                predictor_blocks[predictor_ref],
                                f"predictor block {predictor_ref}",
                            )["columns"],
                            f"predictor block {predictor_ref} columns",
                        )
                    )
                ),
                "person_outputs": [
                    target["name"] for target in targets if target["entity"] == "person"
                ],
                "tax_unit_outputs": [
                    target["name"]
                    for target in targets
                    if target["entity"] == "tax_unit"
                ],
            }
        )
        qrf["seed"] = build_model_seed
        tail = deepcopy(
            dict(_mapping_like(binding["capital_gains_tail"], "capital-gains tail"))
        )
        tail["seed"] = build_model_seed
        _require_typed_ref(
            tail.pop("support_contract_ref"),
            _PUF_TAIL_SUPPORT_REF,
            location=f"resource {resource_id} tail support_contract_ref",
        )
        support_role = _spine_support_role(
            spine_document,
            role_id=_PUF_TAIL_SUPPORT_REF["support_role"],
        )
        tail_support = _mapping_like(
            support_role["tail_support"], "spine tail support"
        )
        tail["support_contract"] = deepcopy(tail_support["legacy_contract"])
        soi_bands = deepcopy(
            dict(
                _mapping_like(
                    tail["soi_e19200_agi_bands"],
                    "capital-gains-tail SOI AGI bands",
                )
            )
        )
        runtime_bands = deepcopy(
            dict(
                _mapping_like(
                    soi_bands["runtime_agi_bands"],
                    "capital-gains-tail runtime AGI bands",
                )
            )
        )
        runtime_bands["sha256"] = _canonical_sha256(runtime_bands)
        soi_bands["runtime_agi_bands"] = runtime_bands
        tail["soi_e19200_agi_bands"] = soi_bands
        _require_typed_ref(
            resolution["clone_attachment_ref"],
            _PUF_ATTACHMENT_REF,
            location="resource semantics clone_attachment_ref",
        )
        attachment_role = _spine_support_role(
            spine_document,
            role_id=_PUF_ATTACHMENT_REF["support_role"],
        )
        attachment = _mapping_like(
            attachment_role["attachment"], "spine PUF attachment"
        )
        fraction = _mapping_like(
            attachment["fraction"], "spine PUF attachment fraction"
        )
        seed = _mapping_like(attachment["seed"], "spine PUF attachment seed")
        assembly = _mapping_like(spine_document["assembly"], "spine assembly")
        binding["clone_attachment"] = {
            "fraction": fraction["default"],
            "seed": seed["default"],
            "support_channels": [
                assembly["mass_anchor_channel"],
                attachment_role["id"],
            ],
            "puf_clone_index": attachment_role["clone_index"],
        }
        binding["qrf"] = qrf
        binding["capital_gains_tail"] = tail
    elif resource_kind == "late_transfer_target_bank":
        dynamic_field = deepcopy(
            dict(
                _mapping_like(
                    resource["dynamic_field"],
                    f"late-transfer target-bank dynamic field for {node['name']}",
                )
            )
        )
        derivation = deepcopy(
            dict(
                _mapping_like(
                    dynamic_field["derivation"],
                    f"late-transfer target-bank derivation for {node['name']}",
                )
            )
        )
        derivation["late_producer_dag_sha256"] = schedule_sha256
        derivation["late_producer_schedule_sha256"] = schedule_payload_sha256
        dynamic_field["derivation"] = derivation
        resource["dynamic_field"] = dynamic_field
    elif resource_kind == "post_clone_source_execution_config":
        metadata = _mapping_like(
            graph["resource_semantics"], "producer resource semantics"
        )
        defaults = _mapping_like(
            metadata["source_execution_defaults"], "source execution defaults"
        )
        operator_registry = _post_clone_operator_order(
            spine_document,
            defaults["operator_registry_ref"],
            location="resource semantics source operator_registry_ref",
        )
        time_period = _target_period(
            bundle_document,
            defaults["time_period_ref"],
            location="resource semantics source time_period_ref",
        )
        operator = str(node["name"]).removeprefix("source:")
        if node["name"] != f"source:{operator}":
            raise RuntimeError(f"Malformed source producer name {node['name']!r}.")
        removed = _mapping_like(
            binding["formula_owned_outputs_removed"],
            f"formula-owned outputs for {operator}",
        )
        output_family: dict[str, set[str]] = {}
        for output_value in _array_like(node["outputs"], f"{operator} outputs"):
            output = _mapping_like(output_value, f"{operator} output")
            if not str(output["column"]).startswith("@"):
                output_family.setdefault(str(output["entity"]), set()).add(
                    str(output["column"])
                )
        for entity, columns in removed.items():
            output_family.setdefault(str(entity), set()).update(
                str(column)
                for column in _array_like(columns, f"removed {entity} outputs")
            )
        binding.update(
            {
                "operator": operator,
                "phase": defaults["phase"],
                "operator_registry": operator_registry,
                "declared_output_family": {
                    entity: sorted(columns)
                    for entity, columns in sorted(output_family.items())
                },
                "seed": build_model_seed,
                "time_period": (
                    None
                    if operator == "impute_us_housing_assistance_to_puf_support"
                    else time_period
                ),
            }
        )
        stage_ref = binding.pop("source_stage_ref")
        if stage_ref is None:
            binding["source_stage_spec"] = None
        else:
            ref = _mapping_like(stage_ref, f"source stage ref for {operator}")
            stage_id = str(ref["stage_id"])
            stages = {
                str(stage["stage"]): stage
                for value in _array_like(sources_document["stages"], "source stages")
                for stage in [_mapping_like(value, "source stage")]
            }
            if len(stages) != len(
                _array_like(sources_document["stages"], "source stages")
            ):
                raise RuntimeError("Source document repeats a stage id.")
            if stage_id not in stages:
                raise RuntimeError(
                    f"Source producer {operator!r} references missing stage {stage_id!r}."
                )
            asset = _mapping_like(
                sources_document["stage_asset"], "source document stage asset"
            )
            if asset["id"] != ref["asset_id"]:
                raise RuntimeError(
                    f"Source producer {operator!r} references source-stage asset "
                    f"{ref['asset_id']!r}, but sources declares {asset['id']!r}."
                )
            manifest = _mapping_like(
                sources_document["stage_manifest"], "source stage manifest"
            )
            resolver = _mapping_like(ref["runtime_resolver"], "source runtime resolver")
            stage_row = _mapping_like(stages[stage_id], f"source stage {stage_id}")
            stage = {
                "stage": stage_row["stage"],
                "survey": stage_row["survey"],
                "source": stage_row["source"],
                "grain": stage_row["grain"],
                "artifacts": deepcopy(list(stage_row.get("artifacts", []))),
                "operations": [
                    {
                        "kind": operation["kind"],
                        "parameters": {
                            key: deepcopy(value)
                            for key, value in operation.items()
                            if key != "kind"
                        },
                    }
                    for value in _array_like(
                        stage_row["operations"], f"source stage {stage_id} operations"
                    )
                    for operation in [
                        _mapping_like(value, f"source stage {stage_id} operation")
                    ]
                ],
                "outputs": deepcopy(list(stage_row["outputs"])),
                "nonnegative_outputs": deepcopy(
                    list(stage_row.get("nonnegative_outputs", []))
                ),
                "notes": stage_row.get("notes", ""),
            }
            binding["source_stage_spec"] = {
                "asset": asset["path"],
                "asset_sha256": asset["sha256"],
                "manifest": deepcopy(dict(manifest)),
                "stage_name": stage_id,
                "resolved_stage_spec": stage,
                "resolved_stage_spec_sha256": _canonical_sha256(stage),
                "runtime_stage_spec_resolver": {
                    "module": resolver["module"],
                    "callable": resolver["callable"],
                },
                "runtime_stage_spec_verified": True,
            }
    elif resource_kind == "primary_qrf_checkpoint":
        family = _primary_family_from_document(document)
        target_order = [
            target["name"]
            for target in _array_like(family["targets"], "primary targets")
        ]
        checkpoint = _mapping_like(document["primary_checkpoint"], "primary checkpoint")
        binding.update(
            {
                "checkpoint_schema_version": checkpoint["schema_version"],
                "target_order": target_order,
                "target_order_sha256": _canonical_sha256(target_order),
            }
        )
    elif resource_kind == "acs_pums_earnings_universe_execution_config":
        identity = deepcopy(
            dict(
                _mapping_like(
                    binding["contract_identity"],
                    "ACS earnings-universe contract identity",
                )
            )
        )
        identity["sha256"] = _canonical_sha256(identity)
        binding["contract_identity"] = identity

    resource["binding"] = binding
    return resource_id, resource


def _derive_canonical_schedule(
    document: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Compile and validate RAW dependencies and family-owned outputs."""

    graph = _mapping_like(document["producer_graph"], "producer graph")
    nodes = [
        _mapping_like(value, "producer graph node")
        for value in _array_like(graph["nodes"], "producer graph nodes")
    ]
    by_name = {str(node["name"]): node for node in nodes}
    if len(by_name) != len(nodes):
        raise RuntimeError("Producer graph repeats a node name.")
    if any(node["id"] != node["name"] for node in nodes):
        raise RuntimeError("Producer graph node ids must equal legacy names.")
    external = set(
        str(stage) for stage in _array_like(graph["external_stages"], "external stages")
    )
    if external & set(by_name):
        raise RuntimeError("Producer and external stage ids overlap.")
    scope_coverage = _mapping_like(graph["scope_coverage"], "scope coverage")
    declared_scopes = _mapping_like(scope_coverage["declared"], "declared scopes")
    node_rank = {str(node["name"]): index for index, node in enumerate(nodes)}
    compiled_outputs = _compile_node_outputs(document, graph)
    edges: set[tuple[str, str]] = set()
    contracts = []
    predecessors = {name: set() for name in by_name}
    proof_nodes: list[dict[str, object]] = []
    for name in sorted(by_name):
        node = by_name[name]
        outputs = [
            {
                "entity": output["entity"],
                "column": output["column"],
                "coverage_scope": output["coverage_scope"],
            }
            for value in compiled_outputs[name]
            for output in [_mapping_like(value, f"{name} output")]
        ]
        expected_write_scopes = [
            _write_scope(
                producer=name,
                output=output,
                ownership_rows=_array_like(
                    graph["ownership_matrix"], "overlap ownership"
                ),
            )
            for output in outputs
        ]
        proof_node = deepcopy(dict(node))
        proof_node["write_scopes"] = expected_write_scopes
        proof_nodes.append(proof_node)
        inputs = deepcopy(list(_array_like(node["inputs"], f"{name} inputs")))
        for value in inputs:
            item = _mapping_like(value, f"{name} input")
            producer_name = str(item["producing_stage"])
            if producer_name in external:
                continue
            if producer_name not in by_name:
                raise RuntimeError(
                    f"Producer node {name!r} references unknown stage {producer_name!r}."
                )
            producer_outputs = [
                _mapping_like(value, f"{producer_name} output")
                for value in _array_like(
                    compiled_outputs[producer_name], f"{producer_name} outputs"
                )
                if _mapping_like(value, f"{producer_name} output")["entity"]
                == item["entity"]
                and _mapping_like(value, f"{producer_name} output")["column"]
                == item["column"]
            ]
            required_scope = str(item["required_scope"])
            if not any(
                required_scope
                in set(
                    _array_like(
                        declared_scopes.get(
                            str(output["coverage_scope"]),
                            [output["coverage_scope"]],
                        ),
                        "covered scopes",
                    )
                )
                for output in producer_outputs
            ):
                raise RuntimeError(
                    f"Producer input {name}/{item['entity']}.{item['column']} has "
                    "no scope-compatible producing output."
                )
            edges.add((producer_name, name))
            predecessors[name].add(producer_name)
        contracts.append(
            {
                "name": name,
                "kind": node["kind"],
                "inputs": inputs,
                "outputs": outputs,
            }
        )
    adjacency = {name: set() for name in by_name}
    indegree = {name: 0 for name in by_name}
    for producer, consumer in edges:
        adjacency[producer].add(consumer)
        indegree[consumer] += 1
    remaining = set(by_name)
    waves: list[list[str]] = []
    while remaining:
        ready = sorted(
            (name for name in remaining if indegree[name] == 0),
            key=node_rank.__getitem__,
        )
        if not ready:
            raise RuntimeError("Producer graph contains a dependency cycle.")
        waves.append(ready)
        remaining.difference_update(ready)
        for producer in ready:
            for consumer in adjacency[producer]:
                indegree[consumer] -= 1
    order = [name for wave in waves for name in wave]
    sorted_edges = [
        list(edge)
        for edge in sorted(
            edges,
            key=lambda edge: (node_rank[edge[0]], node_rank[edge[1]]),
        )
    ]
    _incomparable_proof(
        nodes=proof_nodes,
        edges=sorted_edges,
        scope_coverage=scope_coverage,
    )
    payload = {
        "schema_version": graph["graph_schema_version"],
        "external_stages": sorted(external),
        "scope_coverage": deepcopy(dict(scope_coverage)),
        "contracts": contracts,
        "edges": sorted_edges,
        "waves": waves,
        "order": order,
    }
    return payload, _canonical_sha256(payload)


def _project_resource_semantics(
    document: Mapping[str, object],
    *,
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
) -> dict[str, object]:
    graph = _mapping_like(document["producer_graph"], "producer graph")
    late_families = {
        str(family["runtime_name"]): family
        for value in _array_like(document["families"], "imputation families")
        for family in [_mapping_like(value, "imputation family")]
        if family.get("stage") == "late_producer_dag"
    }
    nodes_by_name = {
        str(node["name"]): node
        for value in _array_like(graph["nodes"], "producer graph nodes")
        for node in [_mapping_like(value, "producer graph node")]
    }
    metadata = _mapping_like(graph["resource_semantics"], "producer resource semantics")
    compiled_schedule, schedule_sha256 = _derive_canonical_schedule(document)
    ownership = _project_overlap_ownership(graph)
    schedule_receipt = _project_late_schedule(document, graph, ownership)
    producers = []
    for producer_name in _array_like(
        compiled_schedule["order"], "compiled producer graph order"
    ):
        node = nodes_by_name[str(producer_name)]
        resources = dict(
            _inflate_virtual_resource(
                document=document,
                node=node,
                resource_row=_mapping_like(resource, "virtual resource"),
                late_families=late_families,
                sources_document=sources_document,
                spine_document=spine_document,
                bundle_document=bundle_document,
                schedule_sha256=schedule_sha256,
                schedule_payload_sha256=str(schedule_receipt["payload_sha256"]),
            )
            for resource in _array_like(
                node["virtual_resources"], "node virtual resources"
            )
        )
        producers.append(
            {
                "producer": node["name"],
                "kind": node["kind"],
                "resources": resources,
            }
        )
    primary = next(
        producer for producer in producers if producer["producer"] == "primary_puf_qrf"
    )
    primary_binding = primary["resources"]["tax_unit.@primary_puf_execution_config"][
        "binding"
    ]
    primary_binding["qrf"]["worker_execution"] = _resolve_worker_execution(
        _mapping_like(
            primary_binding["qrf"]["worker_execution"],
            "primary-QRF worker template",
        )
    )
    result = {
        "artifact_kind": metadata["artifact_kind"],
        "schema_version": metadata["schema_version"],
        "producer_schedule_sha256": schedule_sha256,
        "producer_schedule_payload_sha256": schedule_receipt["payload_sha256"],
        "producer_count": len(producers),
        "producers": producers,
    }
    result["sha256"] = _canonical_sha256(result)
    return result


def _project_overlap_ownership(
    graph: Mapping[str, object],
) -> dict[str, object]:
    result = deepcopy(
        dict(_mapping_like(graph["ownership_contract"], "ownership contract"))
    )
    ownership = deepcopy(list(graph["ownership_matrix"]))
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for value in ownership:
        row = _mapping_like(value, "overlap ownership row")
        grouped.setdefault((str(row["entity"]), str(row["target"])), []).append(row)
    targets = []
    # The matrix is the sole authority for both membership and legacy row
    # order.  Preserve each target's first appearance: sorting here changes
    # the byte-attested overlap receipt and therefore the schedule payload.
    for (entity, target), rows in grouped.items():
        actions = [
            _mapping_like(value, "overlap producer action")
            for row in rows
            for value in _array_like(row["producer_actions"], "producer actions")
        ]
        source_producers = {
            str(action["producer"])
            for action in actions
            if str(action["producer"]).startswith("source:")
        }
        transfer_producers = {
            str(action["producer"])
            for action in actions
            if str(action["producer"]).startswith("transfer:")
        }
        if len(source_producers) != 1 or len(transfer_producers) != 1:
            raise RuntimeError(
                f"Ownership rows for {entity}.{target} do not identify one source "
                "and transfer producer."
            )
        source_actions = {
            str(action["action"])
            for action in actions
            if str(action["producer"]) in source_producers
        }
        if "consume_only_byte_exact_noop" in source_actions:
            source_touch = "consume_only_byte_exact_noop"
        elif "final_write" in source_actions:
            source_touch = "persisted_owner_last_write"
        else:
            raise RuntimeError(
                f"Ownership rows for {entity}.{target} have no reviewed source touch."
            )
        targets.append(
            {
                "entity": entity,
                "target": target,
                "source_producer": next(iter(source_producers)),
                "source_touch": source_touch,
                "transfer_producer": next(iter(transfer_producers)),
            }
        )
    result["targets"] = targets
    result["ownership"] = ownership
    result["sha256"] = _canonical_sha256(result)
    return result


def _derive_input_inventory(
    node: Mapping[str, object],
    *,
    operator: str,
    default_scope: str,
) -> dict[str, object]:
    requirements = []
    node_name = str(node["name"])
    for value in _array_like(node["inputs"], f"{node_name} inputs"):
        input_row = _mapping_like(value, f"{node_name} input")
        column = str(input_row["column"])
        if not column.startswith("@effective:"):
            continue
        label = column.removeprefix("@effective:")
        optional_receipt = f"optional_input:{node_name}:{label}"
        absence_receipts = list(
            _array_like(
                input_row["tolerated_absence_receipts"],
                f"{node_name}/{label} tolerated absence receipts",
            )
        )
        if absence_receipts not in ([], [optional_receipt]):
            raise RuntimeError(
                f"Effective input {node_name}/{label} has unsupported absence receipts."
            )
        required_scope = str(input_row["required_scope"])
        requirements.append(
            {
                "label": label,
                "optional": absence_receipts == [optional_receipt],
                "required_scope": (
                    None if required_scope == default_scope else required_scope
                ),
                "alternatives": deepcopy(list(input_row["alternatives"])),
            }
        )
    requirements.sort(key=lambda row: str(row["label"]))
    return {"operator": operator, "requirements": requirements}


def _derive_input_inventories(graph: Mapping[str, object]) -> dict[str, object]:
    inventories: dict[str, object] = {"sources": [], "transfers": []}
    for value in _array_like(graph["nodes"], "producer graph nodes"):
        node = _mapping_like(value, "producer graph node")
        name = str(node["name"])
        kind = str(node["kind"])
        if kind == "acs_earnings_universe":
            inventories["acs_earnings_universe"] = _derive_input_inventory(
                node, operator=name, default_scope="acs_source"
            )
        elif kind == "primary_puf":
            inventories["primary_puf"] = _derive_input_inventory(
                node, operator=name, default_scope="whole_pool"
            )
        elif kind == "post_clone_source":
            if not name.startswith("source:"):
                raise RuntimeError(f"Source producer {name!r} lacks the source: prefix.")
            inventories["sources"].append(
                _derive_input_inventory(
                    node,
                    operator=name.removeprefix("source:"),
                    default_scope="asec_source",
                )
            )
        elif kind == "late_transfer":
            inventories["transfers"].append(
                _derive_input_inventory(
                    node, operator=name, default_scope="whole_pool"
                )
            )
    if set(inventories) != {
        "acs_earnings_universe",
        "primary_puf",
        "sources",
        "transfers",
    }:
        raise RuntimeError("Producer nodes do not define all legacy input inventories.")
    for key in ("sources", "transfers"):
        inventories[key].sort(key=lambda row: str(row["operator"]))
    return inventories


def _derive_transfer_groups(document: Mapping[str, object]) -> list[dict[str, object]]:
    groups = []
    for value in _array_like(document["families"], "imputation families"):
        family = _mapping_like(value, "imputation family")
        if family.get("stage") != "late_producer_dag":
            continue
        entities = list(_array_like(family["entities"], "late family entities"))
        if len(entities) != 1:
            raise RuntimeError(f"Late family {family['id']!r} must have one entity.")
        entity = str(entities[0])
        prefix = f"late/{entity}/"
        family_id = str(family["id"])
        if not family_id.startswith(prefix):
            raise RuntimeError(f"Late family {family_id!r} does not match {entity!r}.")
        groups.append(
            {
                "name": family["runtime_name"],
                "entity": entity,
                "family": family_id.removeprefix(prefix),
                "targets": [
                    _mapping_like(target, "late family target")["name"]
                    for target in _array_like(family["targets"], "late family targets")
                ],
            }
        )
    return groups


def _project_late_schedule(
    document: Mapping[str, object],
    graph: Mapping[str, object],
    ownership: Mapping[str, object],
) -> dict[str, object]:
    inventories = _derive_input_inventories(graph)
    compiled_schedule, schedule_sha256 = _derive_canonical_schedule(document)
    transfers = deepcopy(list(inventories["transfers"]))
    sources = deepcopy(list(inventories["sources"]))
    groups = _derive_transfer_groups(document)
    payload = {
        "schema_version": graph["schedule_payload_schema_version"],
        "overlap_ownership": deepcopy(dict(ownership)),
        "execution_receipt_contract": deepcopy(graph["execution_receipt_contract"]),
        "schedule_sha256": schedule_sha256,
        "external_stages": deepcopy(compiled_schedule["external_stages"]),
        "order": deepcopy(compiled_schedule["order"]),
        "waves": deepcopy(compiled_schedule["waves"]),
        "edges": deepcopy(compiled_schedule["edges"]),
        "transfer_groups": groups,
        "source_input_inventories": sources,
        "primary_puf_input_inventory": deepcopy(inventories["primary_puf"]),
        "acs_earnings_universe_input_inventory": deepcopy(
            inventories["acs_earnings_universe"]
        ),
        "transfer_input_inventories": transfers,
    }
    payload_sha256 = _canonical_sha256(payload)
    return {
        **payload,
        "payload_sha256": payload_sha256,
        "producer_count": len(graph["nodes"]),
        "source_producer_count": len(sources),
        "transfer_group_count": len(groups),
        "transfer_target_count": sum(len(group["targets"]) for group in groups),
        "status": "derived_and_import_validated",
    }


def _assert_compiler_semantics(document: Mapping[str, object]) -> None:
    """Consume compiler-only references fail-closed."""

    graph = _mapping_like(document["producer_graph"], "producer graph")
    node_ids = {
        str(node["id"])
        for value in _array_like(graph["nodes"], "producer graph nodes")
        for node in [_mapping_like(value, "producer graph node")]
    }
    models = _mapping_like(document["models"], "imputation models")
    profiles = _mapping_like(
        _mapping_like(document["transfer_execution"], "transfer execution")["profiles"],
        "transfer profiles",
    )
    primary = _primary_family_from_document(document)
    if primary["execution_contract"] not in node_ids:
        raise RuntimeError(
            "Primary family execution_contract must resolve to its producer node."
        )
    for value in _array_like(document["families"], "imputation families"):
        family = _mapping_like(value, "imputation family")
        if family["model"] not in models:
            raise RuntimeError(
                f"Imputation family {family['id']!r} references an unknown model."
            )
        if (
            family["stage"] != "primary_puf_qrf"
            and family["execution_contract"] not in profiles
        ):
            raise RuntimeError(
                f"Imputation family {family['id']!r} references an unknown transfer profile."
            )
    derive_primary_effective_predictor_tuples(document)


def derive_primary_effective_predictor_tuples(
    document: Mapping[str, object],
) -> list[dict[str, object]]:
    """Compile the primary chain from its sole predictor block and target order."""

    primary = _primary_family_from_document(document)
    if primary.get("chaining") != "base_plus_preceding_declared_targets":
        raise RuntimeError("Primary family has an unsupported chaining contract.")
    predictor_blocks = _mapping_like(document["predictor_blocks"], "predictor blocks")
    predictor_refs = _array_like(primary["predictors"], "primary predictors")
    if len(predictor_refs) != 1:
        raise RuntimeError(
            "Primary family must reference exactly one base predictor block."
        )
    predictor_ref = str(predictor_refs[0])
    if predictor_ref not in predictor_blocks:
        raise RuntimeError(
            f"Primary family references missing predictor block {predictor_ref!r}."
        )
    base = list(
        _array_like(
            _mapping_like(
                predictor_blocks[predictor_ref], "primary predictor block"
            )["columns"],
            "primary predictor columns",
        )
    )
    preceding: list[object] = []
    result = []
    for value in _array_like(primary["targets"], "primary targets"):
        target = _mapping_like(value, "primary target")
        result.append(
            {
                "target": target["name"],
                "entity": target["entity"],
                "predictors": [*base, *preceding],
            }
        )
        preceding.append(target["name"])
    return result


def project_imputation_legacy_payloads(
    document: Mapping[str, object],
    *,
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct every constants-era imputation identity from typed fields."""

    _assert_compiler_semantics(document)
    gap_fill_plan = _project_gap_fill_plan(document)
    graph = _mapping_like(document["producer_graph"], "producer graph")
    ownership = _project_overlap_ownership(graph)
    primary_family = _primary_family_from_document(document)
    primary_targets = [
        _mapping_like(value, "primary target")
        for value in _array_like(primary_family["targets"], "primary targets")
    ]
    target_order = [target["name"] for target in primary_targets]
    predictor_blocks = _mapping_like(document["predictor_blocks"], "predictor blocks")
    transfer_execution = _mapping_like(
        document["transfer_execution"], "transfer execution"
    )
    profiles = _mapping_like(transfer_execution["profiles"], "transfer profiles")
    early_contracts = []
    for direction in gap_fill_plan:
        ordered_targets = [
            target
            for entity_families in direction["target_families"].values()
            for targets in entity_families.values()
            for target in targets
        ]
        profile_id = "acs_transfer_early"
        early_contracts.append(
            {
                "id": f"early/{direction['name']}",
                "direction": direction["name"],
                "derive_schedule_d": profiles[profile_id]["derive_schedule_d"],
                "ordered_targets": ordered_targets,
                "identity": _project_transfer_execution_identity(
                    transfer_execution,
                    predictor_blocks,
                    profile_id=profile_id,
                    targets=ordered_targets,
                ),
            }
        )
    late_contracts = []
    for value in _array_like(document["families"], "imputation families"):
        family = _mapping_like(value, "imputation family")
        if family.get("stage") != "late_producer_dag":
            continue
        ordered_targets = [
            target["name"]
            for target in _array_like(family["targets"], "late family targets")
        ]
        profile_id = str(family["execution_contract"])
        late_contracts.append(
            {
                "id": family["id"],
                "producer": family["runtime_name"],
                "derive_schedule_d": profiles[profile_id]["derive_schedule_d"],
                "ordered_targets": ordered_targets,
                "identity": _project_transfer_execution_identity(
                    transfer_execution,
                    predictor_blocks,
                    profile_id=profile_id,
                    targets=ordered_targets,
                ),
            }
        )
    checkpoint = _mapping_like(document["primary_checkpoint"], "primary checkpoint")
    predictor_ref = str(
        _array_like(primary_family["predictors"], "primary predictors")[0]
    )
    return {
        "gap_fill_plan": gap_fill_plan,
        "gap_fill_producer_schedule_receipt": _project_gap_fill_schedule(
            document, gap_fill_plan
        ),
        "primary_qrf": {
            "predictors": deepcopy(predictor_blocks[predictor_ref]["columns"]),
            "person_outputs": [
                target["name"]
                for target in primary_targets
                if target["entity"] == "person"
            ],
            "tax_unit_outputs": [
                target["name"]
                for target in primary_targets
                if target["entity"] == "tax_unit"
            ],
            "target_order": target_order,
            "target_order_sha256": _canonical_sha256(target_order),
            "target_order_digest_rule": checkpoint["target_order_digest_rule"],
            "checkpoint_schema_version": checkpoint["schema_version"],
        },
        "late_producer_schedule_receipt": _project_late_schedule(
            document, graph, ownership
        ),
        "late_producer_resource_semantics": _project_resource_semantics(
            document,
            sources_document=sources_document,
            spine_document=spine_document,
            bundle_document=bundle_document,
        ),
        "overlap_ownership": ownership,
        "transfer_execution_contract_identities": {
            "base": _project_transfer_execution_identity(
                transfer_execution,
                predictor_blocks,
                profile_id="acs_transfer_default",
                targets=[],
            ),
            "early": early_contracts,
            "late": late_contracts,
        },
    }


def _split_after() -> list[dict[str, object]]:
    late_batches = [
        group
        for group in CANONICAL_US_LATE_TRANSFER_GROUPS
        if group.entity == "person"
        and group.family.startswith("puf_tax_itemization__batch_")
    ]
    if [len(group.targets) for group in late_batches] != [8, 8, 8, 8, 5]:
        raise RuntimeError("Late PUF itemization is no longer the reviewed 5x37 split.")
    result = [
        {
            "family": f"late/{group.entity}/{group.family}",
            "after_target": group.targets[-1],
            "reason": (
                "Declared production memory boundary preserving the canonical "
                "five-batch late PUF itemization partition."
            ),
        }
        for group in late_batches[:-1]
    ]
    early_puf = next(
        family
        for direction in stacked_gap_fill_plan()
        for entity, families in direction.target_families.items()
        for name, family in families.items()
        if entity == "person" and name == "puf_tax_itemization"
    )
    result.insert(
        0,
        {
            "family": "early/asec_survey_to_acs/person/puf_tax_itemization",
            "after_target": early_puf[DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT - 1],
            "reason": (
                "Declared eight-target QRF memory boundary matching the "
                "constants-era ACS transfer maximum."
            ),
        },
    )
    return result


def _assert_invariants(
    result: Mapping[str, object],
    *,
    expected_legacy_payloads: Mapping[str, object],
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
) -> None:
    families: Sequence[Mapping[str, object]] = result["families"]
    by_stage: dict[str, list[Mapping[str, object]]] = {}
    for family in families:
        by_stage.setdefault(str(family["stage"]), []).append(family)
    expected = {
        "gap_fill_stacked_spine": (13, 48),
        "primary_puf_qrf": (1, 65),
        "late_producer_dag": (19, 70),
    }
    expected_dtypes = {
        "gap_fill_stacked_spine": {"bool": 20, "float": 27, "int": 1},
        "primary_puf_qrf": {"bool": 8, "float": 55, "int": 2},
        "late_producer_dag": {"bool": 17, "float": 50, "int": 1, "str": 2},
    }
    for stage, (family_count, target_count) in expected.items():
        rows = by_stage.get(stage, [])
        observed_targets = sum(len(row["targets"]) for row in rows)
        if (len(rows), observed_targets) != (family_count, target_count):
            raise RuntimeError(
                f"US imputation {stage} expected {family_count} families / "
                f"{target_count} targets, got {len(rows)} / {observed_targets}."
            )
        observed_dtypes: dict[str, int] = {}
        for row in rows:
            for target in row["targets"]:
                dtype = str(target["dtype"])
                observed_dtypes[dtype] = observed_dtypes.get(dtype, 0) + 1
        if observed_dtypes != expected_dtypes[stage]:
            raise RuntimeError(
                f"US imputation {stage} target dtypes changed: {observed_dtypes}."
            )
    graph = result["producer_graph"]
    compiled_schedule, _ = _derive_canonical_schedule(result)
    if (
        len(graph["nodes"]),
        len(compiled_schedule["edges"]),
        len(compiled_schedule["waves"]),
        len(graph["ownership_matrix"]),
    ) != (38, 71, 6, 18):
        raise RuntimeError("US producer graph count invariant changed.")
    input_count = sum(len(node["inputs"]) for node in graph["nodes"])
    authored_output_count = sum(len(node["outputs"]) for node in graph["nodes"])
    compiled_outputs = _compile_node_outputs(result, graph)
    compiled_output_count = sum(len(rows) for rows in compiled_outputs.values())
    primary_node = next(
        node for node in graph["nodes"] if node["name"] == "primary_puf_qrf"
    )
    late_authored_output_count = sum(
        len(node["outputs"])
        for node in graph["nodes"]
        if node["kind"] == "late_transfer"
    )
    tolerated_receipts = [
        receipt
        for node in graph["nodes"]
        for requirement in node["inputs"]
        for receipt in requirement["tolerated_absence_receipts"]
    ]
    if (
        input_count,
        authored_output_count,
        compiled_output_count,
        len(primary_node["outputs"]),
        late_authored_output_count,
        len(tolerated_receipts),
    ) != (2742, 92, 227, 35, 0, 212):
        raise RuntimeError("US producer graph input/output/absence counts changed.")
    if len(set(tolerated_receipts)) != len(tolerated_receipts):
        raise RuntimeError("US producer graph absence receipt IDs are not unique.")
    waived = {
        target["name"]
        for family in families
        for target in family["targets"]
        if "waiver" in target
    }
    if waived != _PARTICIPATION_TARGETS:
        raise RuntimeError(
            "US F-P waiver coverage changed; "
            f"missing={sorted(_PARTICIPATION_TARGETS - waived)}, "
            f"extra={sorted(waived - _PARTICIPATION_TARGETS)}."
        )
    if set(_PARTICIPATION_TARGET_CONCEPTS) != _PARTICIPATION_TARGETS:
        raise RuntimeError(
            "US target-specific concept declarations do not exactly cover the "
            "participation-target inventory."
        )
    for target, required in _PARTICIPATION_TARGET_CONCEPTS.items():
        if not required:
            raise RuntimeError(
                f"US participation target {target!r} has no required concepts."
            )
        unknown = sorted(set(required) - set(_CONCEPTS))
        if unknown:
            raise RuntimeError(
                f"US participation target {target!r} references unknown concepts "
                f"{unknown}."
            )
    waiver_records = _array_like(result["waiver_records"], "waiver_records")
    if len(waiver_records) != 1:
        raise RuntimeError("US imputation must have exactly one F-P waiver record.")
    waiver = _mapping_like(waiver_records[0], "waiver_records/0")
    expected_missing = _missing_participation_concepts(
        families=families,
        predictor_blocks=_mapping_like(result["predictor_blocks"], "predictor_blocks"),
    )
    if waiver.get("missing_concepts_by_target") != expected_missing:
        raise RuntimeError(
            "US F-P waiver does not record every missing concept exactly."
        )
    expected_required = sorted(
        {concept for concepts in expected_missing.values() for concept in concepts}
    )
    if waiver.get("requires_concepts") != expected_required:
        raise RuntimeError("US F-P waiver concept inventory is not the exact union.")
    projected = project_imputation_legacy_payloads(
        result,
        sources_document=sources_document,
        spine_document=spine_document,
        bundle_document=bundle_document,
    )
    model = _mapping_like(
        _mapping_like(result["models"], "imputation models")["regime_gated_qrf"],
        "regime-gated QRF model",
    )
    n_estimators = int(
        _mapping_like(model["params"], "regime-gated QRF parameters")["n_estimators"]
    )
    attachment_role = _spine_support_role(
        spine_document,
        role_id=_PUF_ATTACHMENT_REF["support_role"],
    )
    attachment = _mapping_like(
        attachment_role["attachment"], "spine PUF attachment"
    )
    attachment_fraction = _mapping_like(
        attachment["fraction"], "spine PUF attachment fraction"
    )["default"]
    attachment_seed = _mapping_like(
        attachment["seed"], "spine PUF attachment seed"
    )["default"]
    build_model_seed = _build_model_seed_default(
        _mapping_like(
            _mapping_like(
                _mapping_like(result["producer_graph"], "producer graph")[
                    "resource_semantics"
                ],
                "resource semantics",
            )["resolution"],
            "resource resolution",
        )["build_model_seed_ref"],
        location="resource semantics build_model_seed_ref",
    )
    live_resource_semantics = _json_ready(
        stacked_late_producer_resource_semantics_receipt(
            clone_attachment_fraction=attachment_fraction,
            clone_attachment_seed=attachment_seed,
            primary_seed=build_model_seed,
            primary_n_estimators=n_estimators,
            transfer_seed=build_model_seed,
            transfer_n_estimators=n_estimators,
            transfer_max_targets_per_fit=(DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT),
        )
    )
    expected_resource_keys = {
        "artifact_kind",
        "producer_count",
        "producer_schedule_payload_sha256",
        "producer_schedule_sha256",
        "producers",
        "schema_version",
        "sha256",
    }
    if set(live_resource_semantics) != expected_resource_keys:
        raise RuntimeError("Live resource-semantics receipt keys changed.")
    if live_resource_semantics["sha256"] != _canonical_sha256(
        {
            key: value
            for key, value in live_resource_semantics.items()
            if key != "sha256"
        }
    ):
        raise RuntimeError("Live resource-semantics SHA is not canonical.")
    if projected["late_producer_resource_semantics"] != live_resource_semantics:
        raise RuntimeError(
            "Resolved resource-semantics projection differs from the live receipt."
        )
    execution_contracts = projected["transfer_execution_contract_identities"]
    if (len(execution_contracts["early"]), len(execution_contracts["late"])) != (
        2,
        19,
    ):
        raise RuntimeError("US transfer execution-contract coverage changed.")
    expected_with_resources = {
        **expected_legacy_payloads,
        "late_producer_resource_semantics": live_resource_semantics,
    }
    if projected != expected_with_resources:
        changed = sorted(
            key
            for key in set(projected) | set(expected_with_resources)
            if projected.get(key) != expected_with_resources.get(key)
        )
        raise RuntimeError(
            "Typed imputation projection differs from constants-era authority: "
            f"{changed}."
        )


def build_imputation() -> dict[str, object]:
    """Build the complete constants-era US imputation declaration."""

    from tools.us_bundle_generation.core import build_bundle, build_sources, build_spine

    metadata = PolicyEngineUSVariableMetadataIndex()
    sources_document = build_sources()
    spine_document = build_spine()
    bundle_document = build_bundle()
    base_transfer_contract = _json_ready(acs_transfer_execution_contract_identity())
    resource_semantics = _portable_resource_semantics(n_estimators=DEFAULT_N_ESTIMATORS)
    schedule_receipt = _json_ready(us_late_producer_schedule_receipt())
    ownership = _json_ready(us_late_overlap_ownership_receipt())
    gap_fill_schedule = _json_ready(stacked_gap_fill_producer_schedule_receipt())

    early_families = _early_families(
        metadata,
        gap_fill_schedule=gap_fill_schedule,
    )
    primary_family = _primary_family(metadata)
    late_families = _late_families(
        metadata,
        resource_semantics=resource_semantics,
    )

    gap_fill_plan = _json_ready(stacked_gap_fill_plan())
    predictor_blocks = _predictor_blocks(base_transfer_contract)
    families = [*early_families, primary_family, *late_families]
    missing_concepts_by_target = _missing_participation_concepts(
        families=families,
        predictor_blocks=predictor_blocks,
    )
    result: dict[str, object] = {
        "predictor_blocks": predictor_blocks,
        "models": {
            "regime_gated_qrf": {
                "kernel": "kernel:regime_gated_qrf",
                "params": {
                    "n_estimators": DEFAULT_N_ESTIMATORS,
                    "max_samples_leaf": None,
                    "zero_atol": DEFAULT_ZERO_ATOL,
                },
            }
        },
        "chaining": {
            "order": "declared",
            "splits": "declared_only",
            "split_after": _split_after(),
            "memory_policy": "release_after_draw",
            "keep_together": [["ssn_card_type", "immigration_status_str"]],
        },
        # These are required eligibility semantics, not newly-added runtime
        # predictors.  The waiver below attests the exact constants-era gaps.
        "concepts": {
            concept_id: list(columns) for concept_id, columns in _CONCEPTS.items()
        },
        "waiver_records": [
            {
                "id": _F_P_WAIVER_ID,
                "code": "F-P",
                "marker": _F_P_WAIVER,
                "reason": "eligibility_concepts_absent",
                "coverage_status": (
                    "required_concepts_not_covered_by_current_predictors"
                ),
                "targets": sorted(_PARTICIPATION_TARGETS),
                "requires_concepts": sorted(
                    {
                        concept
                        for concepts in missing_concepts_by_target.values()
                        for concept in concepts
                    }
                ),
                "missing_concepts_by_target": missing_concepts_by_target,
            }
        ],
        "transfer_execution": _normalise_transfer_execution(base_transfer_contract),
        "gap_fill_schedule": _normalise_gap_fill_schedule(gap_fill_schedule),
        "primary_checkpoint": {
            "schema_version": PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
            "target_order_digest_rule": "sha256(compact_json_array)",
        },
        "families": families,
        "producer_graph": _producer_graph(
            schedule_receipt=schedule_receipt,
            resource_semantics=resource_semantics,
            ownership=ownership,
            families=families,
        ),
    }

    semantics_by_producer = {
        row["producer"]: row for row in resource_semantics["producers"]
    }
    early_contracts = []
    for direction in stacked_gap_fill_plan():
        ordered_targets = _ordered_direction_targets(direction)
        early_contracts.append(
            {
                "id": f"early/{direction.name}",
                "direction": direction.name,
                "derive_schedule_d": True,
                "ordered_targets": ordered_targets,
                "identity": _json_ready(
                    acs_transfer_execution_contract_identity(
                        targets=ordered_targets,
                        derive_schedule_d=True,
                    )
                ),
            }
        )
    late_contracts = []
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        binding = semantics_by_producer[group.name]["resources"][
            f"{group.entity}.@late_transfer_model_config"
        ]["binding"]
        late_contracts.append(
            {
                "id": f"late/{group.entity}/{group.family}",
                "producer": group.name,
                "derive_schedule_d": False,
                "ordered_targets": list(group.targets),
                "identity": deepcopy(binding["transfer_execution_contract"]),
            }
        )
    expected_legacy_payloads = {
        "gap_fill_plan": gap_fill_plan,
        "gap_fill_producer_schedule_receipt": gap_fill_schedule,
        "primary_qrf": {
            "predictors": list(PUF_TAX_DETAIL_DEFAULT_PREDICTORS),
            "person_outputs": list(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS),
            "tax_unit_outputs": list(PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS),
            "target_order": list(PRIMARY_QRF_TARGET_ORDER),
            "target_order_sha256": PRIMARY_QRF_TARGET_ORDER_SHA256,
            "target_order_digest_rule": "sha256(compact_json_array)",
            "checkpoint_schema_version": (PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION),
        },
        "late_producer_schedule_receipt": schedule_receipt,
        "overlap_ownership": ownership,
        "transfer_execution_contract_identities": {
            "base": base_transfer_contract,
            "early": early_contracts,
            "late": late_contracts,
        },
    }
    _assert_invariants(
        result,
        expected_legacy_payloads=expected_legacy_payloads,
        sources_document=sources_document,
        spine_document=spine_document,
        bundle_document=bundle_document,
    )
    return result


__all__ = [
    "build_imputation",
    "derive_primary_effective_predictor_tuples",
    "project_imputation_legacy_payloads",
]
