#!/usr/bin/env python3
"""Phase-P acceptance gate for the per-family predictor-set change.

The command compares a held baseline stacked-f025 artifact with a candidate
stacked-f025 artifact.  It intentionally does not consume library defaults for
the statistical design: the fold, quantiles, forest, and bootstrap are pinned
in :data:`PHASE_P_CONFIG` below and are copied into every verdict receipt.

The OOS comparison refits one quantile forest per modeled target.  Family
target order comes from each artifact's own execution receipt; preceding
targets in the same bounded chain are supplied as observed (teacher-forced)
predictors.  This makes the baseline CHAMPVA-first order and the candidate
disability-first order part of the measured treatment.  Optional predictor
missingness is handled deterministically with a training-fold median plus an
observed-value indicator, while required-predictor null rows remain ineligible.
The paired estimand is the difference in full-artifact-arm OOS risk: stable
donor ids and bootstrap clusters pair the rows, while each arm retains the
target values and effective importance weights its runtime actually consumed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from quantile_forest import RandomForestQuantileRegressor

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.acs_transfer import (
    AcsPredictorSet,
    _aggregate_optional_to_group,
    _complete_case_target_encodings,
    _complete_numeric_component_sum,
    _complete_target_mask,
    _legacy_predictor_set,
    _transfer_feature_surface,
    declared_acs_transfer_predictor_sets,
)
from microcosm.build.us_runtime.acs_transfer_bank import (
    ACS_TRANSFER_TARGET_BANK_ARTIFACT_KIND,
    ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
    ACS_TRANSFER_TARGET_BANK_RECEIPT_ARTIFACT_KIND,
    ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION,
)
from microcosm.build.us_runtime.stacked_spine import (
    ORIGIN_BATTERY_METRIC_KINDS,
    declared_stacked_post_puf_predictor_sets,
)
from microcosm.fit import QRFChainState
from microcosm.frame import Frame

# ---------------------------------------------------------------------------
# Phase-P statistical configuration.  These are gate authority, not tunables.
# ---------------------------------------------------------------------------

PHASE_P_CONFIG: dict[str, object] = {
    "schema_version": 2,
    "fold": {
        "algorithm": "sha256(namespace + NUL + stable_source_id)",
        "seed": 20260816,
        "fold_count": 5,
        "held_out_fold": 0,
        "unit": "entity_source_id",
    },
    "quantiles": [0.10, 0.25, 0.50, 0.75, 0.90],
    "quantile_loss": {
        "per_row_quantile_aggregation": "unweighted_arithmetic_mean",
        "per_target_holdout_aggregation": "arm_specific_scoring_weighted_mean",
        "paired_delta": "candidate_minus_baseline",
    },
    "optional_missingness": {
        "value": "unweighted_training_fold_median",
        "indicator": "one_observed_value_indicator_per_optional_predictor",
        "all_missing_training_predictor": "drop_and_report",
        "required_missing": "row_ineligible",
    },
    "forest": {
        "weighted_training": "seeded_n_of_n_probability_weighted_bootstrap",
        "seed": 260816,
        "n_estimators": 32,
        "default_quantiles": 0.5,
        "criterion": "squared_error",
        "max_depth": 16,
        "min_samples_split": 40,
        "min_samples_leaf": 20,
        "max_samples_leaf": 256,
        "min_weight_fraction_leaf": 0.0,
        "max_features": 1.0,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": 4,
        "verbose": 0,
        "warm_start": False,
        "ccp_alpha": 0.0,
        "max_samples": 0.8,
        "monotonic_cst": None,
        "predict_interpolation": "linear",
        "predict_weighted_quantile": True,
        "predict_weighted_leaves": False,
        "predict_aggregate_leaves_first": True,
        "minimum_training_rows": 100,
        "minimum_holdout_rows": 20,
    },
    "bootstrap": {
        "method": "paired_cluster_nonparametric_percentile",
        "cluster_assignment": "sha256(seed + NUL + stable_source_id)",
        "seed": 8162026,
        "clusters": 256,
        "replicates": 2000,
        "confidence_level": 0.95,
        "percentile_method": "linear",
    },
    "paired_oos_estimand": {
        "name": "full_artifact_arm_weighted_oos_risk_difference",
        "pairing_authority": "identical_entity_source_id_sets",
        "fold_authority": "shared_deterministic_id_hash",
        "bootstrap_pairing": "shared_cluster_multiplicities",
        "training_weights": "arm_specific_frame_resolve_weights",
        "scoring_weights": "arm_specific_frame_resolve_weights",
        "scoring_outcomes": "arm_specific_simulated_checkpoint_target_values",
        "normalization": "separate_arm_weight_denominators",
        "weight_drift_policy": "allowed_and_reported",
        "outcome_drift_policy": "allowed_and_reported",
        "drift_diagnostics_absolute_tolerance": 1e-10,
    },
    "money_amount_classification": {
        "authority": "explicit_phase_p_family_target_allowlist",
        "changed_target_count": 118,
        "generic_numeric_metric_target_count": 77,
        "money_amount_target_count": 74,
        "money_amount_target_keys": [
            "gap_fill/asec_housing_to_acs/person/housing/pre_subsidy_rent",
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/health_insurance_premiums_without_medicare_part_b",
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/other_medical_expenses",
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/over_the_counter_health_expenses",
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/tax_exempt_private_pension_income",
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/unemployment_compensation",
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/veterans_benefits",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/alimony_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/farm_operations_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/long_term_capital_gains_before_response",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/miscellaneous_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/non_qualified_dividend_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/qualified_dividend_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/rental_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/short_term_capital_gains",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/social_security_dependents",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/social_security_disability",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/social_security_retirement",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/social_security_survivors",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/taxable_interest_income",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/taxable_ira_distributions",
            "gap_fill/asec_survey_to_acs/person/puf_tax_itemization/taxable_private_pension_income",
            "gap_fill/asec_survey_to_acs/person/source_operator_cps_carried/strike_benefits",
            "gap_fill/asec_survey_to_acs/person/source_operator_prior_year_income/self_employment_income_last_year",
            "gap_fill/asec_survey_to_acs/spm_unit/model_required_numeric/spm_unit_pre_subsidy_childcare_expenses",
            "late_transfer/transfer:person/adult_care/pre_subsidy_care_expenses",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/alimony_expense",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/charitable_cash_donations",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/charitable_non_cash_donations",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/home_mortgage_interest",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/long_term_capital_gains_on_collectibles",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/non_sch_d_capital_gains",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/salt_refund_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_1/tax_exempt_interest_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/casualty_loss",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/educator_expense",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/investment_income_elected_form_4952",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/investment_interest_expense",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/qualified_tuition_expenses",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/student_loan_interest",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/traditional_ira_contributions_desired",
            "late_transfer/transfer:person/puf_tax_itemization__batch_2/unreimbursed_business_employee_expenses",
            "late_transfer/transfer:person/puf_tax_itemization__batch_3/estate_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_3/farm_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_3/farm_rent_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_3/partnership_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_3/partnership_self_employment_net_earnings",
            "late_transfer/transfer:person/puf_tax_itemization__batch_3/self_employed_pension_contributions_desired",
            "late_transfer/transfer:person/puf_tax_itemization__batch_4/qualified_bdc_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_4/qualified_reit_and_ptp_income",
            "late_transfer/transfer:person/puf_tax_itemization__batch_5/sstb_self_employment_income_before_lsr",
            "late_transfer/transfer:person/puf_tax_itemization__batch_5/sstb_unadjusted_basis_qualified_property",
            "late_transfer/transfer:person/puf_tax_itemization__batch_5/sstb_w2_wages_from_qualified_business",
            "late_transfer/transfer:person/puf_tax_itemization__batch_5/unadjusted_basis_qualified_property",
            "late_transfer/transfer:person/puf_tax_itemization__batch_5/w2_wages_from_qualified_business",
            "late_transfer/transfer:person/source_operator_child_support/child_support_expense",
            "late_transfer/transfer:person/source_operator_child_support/child_support_received",
            "late_transfer/transfer:person/source_operator_disability_benefits/disability_benefits",
            "late_transfer/transfer:person/source_operator_education_inputs/educational_assistance",
            "late_transfer/transfer:person/source_operator_retirement_contributions/roth_401k_contributions_desired",
            "late_transfer/transfer:person/source_operator_retirement_contributions/roth_ira_contributions_desired",
            "late_transfer/transfer:person/source_operator_retirement_contributions/traditional_401k_contributions_desired",
            "late_transfer/transfer:person/source_operator_retirement_distributions/keogh_distributions",
            "late_transfer/transfer:person/source_operator_retirement_distributions/tax_exempt_ira_distributions",
            "late_transfer/transfer:person/source_operator_retirement_distributions/taxable_401k_distributions",
            "late_transfer/transfer:person/source_operator_retirement_distributions/taxable_403b_distributions",
            "late_transfer/transfer:person/source_operator_retirement_distributions/taxable_sep_distributions",
            "late_transfer/transfer:person/source_operator_workers_compensation/workers_compensation",
            "late_transfer/transfer:spm_unit/source_operator_energy_subsidy/spm_unit_energy_subsidy",
            "late_transfer/transfer:tax_unit/puf_tax_itemization/domestic_production_ald",
            "late_transfer/transfer:tax_unit/puf_tax_itemization/first_home_mortgage_balance",
            "late_transfer/transfer:tax_unit/puf_tax_itemization/first_home_mortgage_interest",
            "late_transfer/transfer:tax_unit/puf_tax_itemization/health_savings_account_ald",
            "late_transfer/transfer:tax_unit/puf_tax_itemization/unrecaptured_section_1250_gain",
        ],
        "generic_numeric_non_money_target_keys": [
            "gap_fill/asec_survey_to_acs/person/model_required_numeric/hours_worked_last_week",
            "gap_fill/asec_survey_to_acs/person/source_operator_hours_worked/weekly_hours_worked_before_lsr",
            "late_transfer/transfer:person/source_operator_weeks_unemployed/weeks_unemployed",
        ],
    },
    "execution_surface": {
        "operational_receipts": {
            "artifact_kind": (
                "populace_us_multispine_pool_stage_checkpoint_operational_receipts"
            ),
            "schema_version": 1,
            "materializer_version": 7,
            "stage": "simulated",
        },
        "early_investment": {
            "person_feature": ("__acs_transfer_interest_dividend_rental_income"),
            "group_feature": "__acs_transfer_group_investment_income_sum",
            "candidate_donor_components": ["INT_VAL", "DIV_VAL", "RNT_VAL"],
            "baseline_donor_source": "unavailable_on_early_execution_surface",
            "candidate_person_donor_source": "complete_numeric_component_sum",
            "candidate_group_projection": (
                "microcosm.build.us_runtime.acs_transfer._aggregate_optional_to_group"
            ),
        },
        "artifact_role_bindings": {
            "baseline": {
                "stacked_authority_version": 10,
                "stacked_authority_sha256": (
                    "f0b676f6508dbf6bb2b787c42e6b85331bacc57c6649ac7ad15fdaa5884a1b2d"
                ),
                "gap_fill_plan_sha256": (
                    "f70bc89cd887504645bd299501e9a63ea23726fb8d9f3fe523092639e251fe0f"
                ),
                "early_target_bank_identity_sha256": {
                    "asec_survey_to_acs": (
                        "4cc8d5ceed38cf61b459a748cee209afabb3a8ccc4cf6cb2edfd089a311aba78"
                    ),
                    "asec_housing_to_acs": (
                        "ba2dd68d0d32ca9bc1eddaf18af64c8bec7e93d649108f398f14806bb42c67e5"
                    ),
                },
            },
            "candidate": {
                "stacked_authority_version": 11,
                "stacked_authority_sha256": (
                    "5a397d602e3a9549c2ce986eab1e098c47886b597b05c892306ef06757950a71"
                ),
                "gap_fill_plan_sha256": (
                    "fdee403f065277448ecd97084c4110cc89304e6586d1717fe60d1669dc5c8af9"
                ),
                "early_target_bank_identity_sha256": {
                    "asec_survey_to_acs": (
                        "26c20db9b43eecac36776e63047c324667b46c7eb7bfba7037d5fa5ab79cb474"
                    ),
                    "asec_housing_to_acs": (
                        "fa4ad7c7d8cc696d4d1ee7d8b9d6018a84a8a14f905e52f5e0fb494e7f47c21e"
                    ),
                },
            },
        },
    },
    "battery_receipt": {
        "top_level_keys": [
            "agreement_gate",
            "artifact_kind",
            "pipeline",
            "publication_run_id",
            "release_id",
            "schema_version",
            "semantic_kind",
            "simulation_ready",
            "terminal_gates",
        ],
        "artifact_kind": "populace_us_multispine_agreement_diagnostics",
        "schema_version": 8,
        "semantic_kind": "stacked_terminal_gates",
        "pipeline": "us-stacked-pool",
        "gate_id": "us_by_origin_battery",
        "gate_keys": ["details", "failures", "passed"],
        "details_keys": [
            "authority",
            "comparisons",
            "declared_plan",
            "declared_target_count",
            "extra_registered_targets",
            "missing_declared_targets",
            "puf_capital_gains_tail_support",
            "registered_joint_target_count",
            "registered_specs",
            "registered_target_count",
            "support_profile",
            "tested_comparisons",
            "tolerances",
            "untestable_comparisons",
        ],
        "declared_plan_id": "stacked_gap_fill_plan",
        "tolerances": {
            "incidence_ratio_bounds": [0.8, 1.25],
            "max_categorical_total_variation_distance": 0.25,
            "max_quantile_envelope_distance": 0.25,
        },
        "monetary_leg_names": ["negative", "positive"],
    },
    "verdict": {
        "ratio_band": [0.8, 1.25],
        "champva_leg": (
            "person/model_required_boolean/"
            "has_champva_health_coverage_at_interview[clone_0]"
        ),
        "champva_minimum_progress_to_band_ceiling": 0.50,
        "champva_progress_anchor": "baseline artifact ratio",
        "money_worsening": (
            "candidate_minus_baseline paired OOS loss CI lower bound > 0"
        ),
        "pass_iff": [
            "a_champva",
            "b_money_oos",
            "c_no_battery_leg_leaves_band",
        ],
    },
}

_CHECKPOINT_ROOT = "_populace_frame_checkpoint"
_CHECKPOINT_METADATA = "metadata_json"
_ASEC_CHANNEL = "asec"
_ACS_CHANNEL = "acs"
_CLONE_ZERO = 0
_PUF_CLONE = 1
_TARGET_METADATA_DATASET = "metadata_json"
_TARGET_RAW_DRAW_DATASET = "raw_draw_bits"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_BATCH_SUFFIX = re.compile(r"__batch_[1-9][0-9]*$")


@dataclass(frozen=True)
class Artifact:
    """Resolved paths and small receipts for one stacked artifact."""

    root: Path
    gates_path: Path
    checkpoint_path: Path
    operational_receipts_path: Path
    gates: dict[str, object]
    checkpoint_metadata: dict[str, object]


@dataclass(frozen=True)
class FamilySpec:
    """One changed family and the exact A/B model declarations."""

    family_id: str
    stage: str
    entity: str
    family: str
    baseline_targets: tuple[str, ...]
    candidate_targets: tuple[str, ...]
    max_targets_per_fit: int
    baseline_predictor_set: AcsPredictorSet
    candidate_predictor_set: AcsPredictorSet

    @property
    def donor_clone_index(self) -> int:
        return _CLONE_ZERO if self.stage == "early_gap_fill" else _PUF_CLONE


@dataclass(frozen=True)
class OOSTargetMeasurement:
    """Internal per-row OOS losses retained until the paired comparison."""

    key: str
    family_id: str
    stage: str
    entity: str
    family: str
    target: str
    target_position: int
    chain_predecessors: tuple[str, ...]
    predictor_set: dict[str, list[str]]
    resolved_predictors: tuple[str, ...]
    unavailable_optional_predictors: tuple[str, ...]
    training_rows: int
    holdout_rows: int
    ids: np.ndarray
    weights: np.ndarray
    truth: np.ndarray
    losses: np.ndarray
    clusters: np.ndarray
    predictor_source_bindings: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EarlyTargetExecution:
    """Authenticated early target-bank evidence for one exported target."""

    family_id: str
    direction: str
    entity: str
    family: str
    target: str
    bank_identity_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    pattern_predictors: tuple[tuple[str, ...], ...]
    resolved_predictors: tuple[str, ...]


@dataclass(frozen=True)
class EarlyExecutionEvidence:
    """Operational evidence binding all early targets for one artifact arm."""

    receipts_path: Path
    receipts_sha256: str
    simulated_checkpoint_sha256: str
    targets: Mapping[tuple[str, str], EarlyTargetExecution]


@dataclass(frozen=True)
class OOSFeatureSurface:
    """Donor-only OOS surface plus its execution-source receipt."""

    features: pd.DataFrame
    required: tuple[str, ...]
    optional: tuple[str, ...]
    source_bindings: dict[str, object]


@dataclass(frozen=True)
class IncidenceMeasurement:
    """Per-row inputs needed for the paired CHAMPVA ratio bootstrap."""

    origin: str
    ids: np.ndarray
    weights: np.ndarray
    values: np.ndarray
    clusters: np.ndarray


@dataclass(frozen=True)
class ArmMeasurement:
    """All retained measurements for one artifact after its Frame is freed."""

    targets: dict[str, OOSTargetMeasurement]
    champva_by_origin: dict[str, IncidenceMeasurement]


class GateMeasurementError(RuntimeError):
    """A contextualized Phase-P measurement failure that must fail closed."""

    def __init__(
        self,
        *,
        arm: str,
        family_id: str,
        stage: str,
        entity: str,
        family: str,
        cause: Exception,
    ) -> None:
        self.context = {
            "arm": arm,
            "family_id": family_id,
            "stage": stage,
            "entity": entity,
            "family": family,
            "cause_type": type(cause).__name__,
            "cause": str(cause),
        }
        super().__init__(
            f"Phase-P {arm} measurement failed for {family_id}: "
            f"{type(cause).__name__}: {cause}"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _bank_mapping_sha256(value: Mapping[str, object]) -> str:
    """Match the target-bank canonical JSON contract exactly."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_bucket(value: object, *, namespace: str, buckets: int) -> int:
    if buckets <= 0:
        raise ValueError("Stable bucket count must be positive.")
    payload = f"{namespace}\0{value!s}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % buckets


def _seed(namespace: str, *parts: object) -> int:
    payload = "\0".join([namespace, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _read_checkpoint_metadata(path: Path) -> dict[str, object]:
    with h5py.File(path, mode="r") as h5:
        dataset = h5[f"{_CHECKPOINT_ROOT}/{_CHECKPOINT_METADATA}"]
        payload = bytes(dataset[...])
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata in {path} is not an object.")
    return value


def resolve_artifact(path: str | Path) -> Artifact:
    """Resolve one directory containing a stacked pool and its checkpoint."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Stacked artifact directory not found: {root}.")
    gate_paths = sorted(root.glob("*.gates.json"))
    if len(gate_paths) != 1:
        raise ValueError(
            f"Stacked artifact {root} must contain exactly one *.gates.json; "
            f"found {[item.name for item in gate_paths]}."
        )
    gates_path = gate_paths[0]
    stem = gates_path.name.removesuffix(".gates.json")
    checkpoint_paths = sorted(
        (root / f"{stem}.checkpoints" / "stacked").glob("*/simulated.checkpoint.h5")
    )
    if len(checkpoint_paths) != 1:
        raise ValueError(
            f"Stacked artifact {root} must contain exactly one "
            f"{stem}.checkpoints/stacked/<identity>/simulated.checkpoint.h5; "
            f"found {checkpoint_paths}."
        )
    gates = json.loads(gates_path.read_text())
    if not isinstance(gates, dict):
        raise ValueError(f"Gate receipt {gates_path} is not a JSON object.")
    checkpoint_path = checkpoint_paths[0]
    return Artifact(
        root=root,
        gates_path=gates_path,
        checkpoint_path=checkpoint_path,
        operational_receipts_path=checkpoint_path.with_suffix(".receipts.json"),
        gates=gates,
        checkpoint_metadata=_read_checkpoint_metadata(checkpoint_path),
    )


def _external_metadata(artifact: Artifact) -> dict[str, object]:
    value = artifact.checkpoint_metadata.get("external_metadata")
    if not isinstance(value, dict):
        raise ValueError(
            f"Checkpoint {artifact.checkpoint_path} lacks object external_metadata."
        )
    return value


def _nested(mapping: Mapping[str, object], *keys: str) -> object:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"Artifact receipt lacks {'.'.join(keys)}.")
        current = current[key]
    return current


def _execution_surface_config() -> Mapping[str, object]:
    value = PHASE_P_CONFIG.get("execution_surface")
    if not isinstance(value, Mapping):  # pragma: no cover - fixed config
        raise TypeError("Phase-P execution_surface config must be an object.")
    return value


def _artifact_role_binding(arm: str) -> Mapping[str, object]:
    roles = _execution_surface_config().get("artifact_role_bindings")
    if not isinstance(roles, Mapping) or arm not in roles:
        raise ValueError(f"Phase-P has no artifact role binding for {arm!r}.")
    binding = roles[arm]
    if not isinstance(binding, Mapping):  # pragma: no cover - fixed config
        raise TypeError(f"Phase-P artifact role {arm!r} must be an object.")
    return binding


def _validate_artifact_role(
    artifact: Artifact,
    *,
    arm: str,
    role_binding: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Bind baseline/candidate semantics to the audited stacked authorities."""

    binding = _artifact_role_binding(arm) if role_binding is None else role_binding
    external = _external_metadata(artifact)
    expected_external = {
        "artifact_kind": "populace_us_multispine_pool_stage_checkpoint",
        "schema_version": 1,
        "materializer_version": 7,
        "stage": "simulated",
    }
    changed = {
        key: external.get(key)
        for key, expected in expected_external.items()
        if external.get(key) != expected
    }
    if changed:
        raise ValueError(
            f"Phase-P {arm} simulated checkpoint envelope changed: {changed}."
        )
    authority = _nested(external, "identity", "stacked_authority")
    if not isinstance(authority, Mapping):
        raise ValueError(f"Phase-P {arm} stacked authority is malformed.")
    expected_sha256 = _require_sha256(
        binding.get("stacked_authority_sha256"),
        label=f"Phase-P {arm} configured stacked authority",
    )
    if (
        authority.get("version") != binding.get("stacked_authority_version")
        or authority.get("sha256") != expected_sha256
        or authority.get("declared_sha256") != expected_sha256
        or authority.get("authority_id") != "us_stacked_spine_authority"
        or authority.get("authority_form") != "CANONICAL"
        or authority.get("declared_authority_form") != "CANONICAL"
        or authority.get("canonical") is not True
        or authority.get("canonical_content") is not True
        or authority.get("canonical_identity") is not True
        or authority.get("digest_matches_declared") is not True
        or authority.get("integrity_valid") is not True
        or authority.get("production_manifest_permitted") is not True
    ):
        raise ValueError(
            f"Phase-P {arm} artifact does not match its audited stacked authority."
        )
    gap_fill = _nested(authority, "components", "gap_fill_plan")
    if not isinstance(gap_fill, Mapping):
        raise ValueError(f"Phase-P {arm} gap-fill authority is malformed.")
    expected_gap = _require_sha256(
        binding.get("gap_fill_plan_sha256"),
        label=f"Phase-P {arm} configured gap-fill plan",
    )
    if (
        gap_fill.get("sha256") != expected_gap
        or gap_fill.get("declared_sha256") != expected_gap
        or gap_fill.get("digest_matches_declared") is not True
        or gap_fill.get("target_count") != 48
        or gap_fill.get("direction_count") != 2
    ):
        raise ValueError(
            f"Phase-P {arm} artifact does not match its audited gap-fill plan."
        )
    return binding


def _early_direction(spec: FamilySpec) -> str:
    pieces = spec.family_id.split("/", 3)
    if (
        spec.stage != "early_gap_fill"
        or len(pieces) != 4
        or pieces[0] != "gap_fill"
        or pieces[2] != spec.entity
        or pieces[3] != spec.family
    ):
        raise ValueError(
            f"Early Phase-P family id {spec.family_id!r} is not direction-bound."
        )
    return pieces[1]


def _string_list(value: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"{label} must be a nonempty list of strings.")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicates.")
    return result


def _read_target_execution(
    path: Path,
    *,
    record: Mapping[str, object],
    bank_identity: Mapping[str, object],
    bank_identity_sha256: str,
) -> tuple[Mapping[str, object], tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Validate one target H5 and its raw-draw/content digests."""

    if not path.is_file():
        raise FileNotFoundError(f"ACS target checkpoint is missing: {path}.")
    expected_file_sha256 = _require_sha256(
        record.get("checkpoint_sha256"),
        label=f"ACS target checkpoint receipt {path}",
    )
    if path.stat().st_size != record.get("size_bytes"):
        raise ValueError(f"ACS target checkpoint size changed: {path}.")
    if _sha256_file(path) != expected_file_sha256:
        raise ValueError(f"ACS target checkpoint SHA-256 changed: {path}.")

    with h5py.File(path, mode="r") as h5:
        if set(h5) != {_TARGET_METADATA_DATASET, _TARGET_RAW_DRAW_DATASET}:
            raise ValueError(f"ACS target checkpoint datasets changed: {path}.")
        metadata_dataset = h5[_TARGET_METADATA_DATASET]
        raw_dataset = h5[_TARGET_RAW_DRAW_DATASET]
        if metadata_dataset.ndim != 1 or metadata_dataset.dtype != np.dtype(np.uint8):
            raise ValueError(f"ACS target metadata dataset changed: {path}.")
        if raw_dataset.ndim != 1 or raw_dataset.dtype != np.dtype("<u8"):
            raise ValueError(f"ACS target raw-draw dataset changed: {path}.")
        metadata_bytes = np.asarray(metadata_dataset, dtype=np.uint8).tobytes()
        raw_rows = int(raw_dataset.shape[0])
        raw_sha256 = hashlib.sha256(
            np.asarray(raw_dataset, dtype=np.dtype("<u8")).tobytes()
        ).hexdigest()
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ACS target metadata JSON is invalid: {path}.") from error
    if not isinstance(metadata, Mapping):
        raise ValueError(f"ACS target metadata must be an object: {path}.")
    expected_metadata_keys = {
        "artifact_kind",
        "schema_version",
        "materializer_version",
        "identity",
        "identity_sha256",
        "target",
        "recipient_rows",
        "raw_draw_sha256",
        "pattern_steps",
        "content_metadata_sha256",
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError(f"ACS target metadata keys changed: {path}.")
    if (
        metadata.get("artifact_kind") != ACS_TRANSFER_TARGET_BANK_ARTIFACT_KIND
        or metadata.get("schema_version") != ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION
        or metadata.get("materializer_version")
        != ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION
        or metadata.get("identity") != bank_identity
        or metadata.get("identity_sha256") != bank_identity_sha256
        or metadata.get("target") != record.get("descriptor")
        or metadata.get("recipient_rows") != raw_rows
        or metadata.get("raw_draw_sha256") != record.get("raw_draw_sha256")
        or metadata.get("raw_draw_sha256") != raw_sha256
        or metadata.get("content_metadata_sha256")
        != record.get("content_metadata_sha256")
    ):
        raise ValueError(f"ACS target checkpoint binding changed: {path}.")
    content_sha256 = _require_sha256(
        metadata.get("content_metadata_sha256"),
        label=f"ACS target content metadata {path}",
    )
    content = {
        key: value
        for key, value in metadata.items()
        if key != "content_metadata_sha256"
    }
    if _bank_mapping_sha256(content) != content_sha256:
        raise ValueError(f"ACS target content metadata digest changed: {path}.")
    _require_sha256(
        metadata.get("raw_draw_sha256"),
        label=f"ACS target raw draw {path}",
    )

    descriptor = metadata.get("target")
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"ACS target descriptor is malformed: {path}.")
    expected_descriptor_keys = {
        "target_index",
        "total_targets",
        "entity",
        "family",
        "family_targets",
        "model_targets",
        "model_target",
        "exported_targets",
    }
    if set(descriptor) != expected_descriptor_keys:
        raise ValueError(f"ACS target descriptor keys changed: {path}.")
    model_targets = _string_list(
        descriptor.get("model_targets"), label=f"ACS model targets {path}"
    )
    _string_list(descriptor.get("family_targets"), label=f"ACS family targets {path}")
    _string_list(
        descriptor.get("exported_targets"), label=f"ACS exported targets {path}"
    )
    model_target = descriptor.get("model_target")
    if model_target not in model_targets:
        raise ValueError(f"ACS model target is absent from its chain: {path}.")

    raw_steps = metadata.get("pattern_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"ACS target pattern_steps are malformed: {path}.")
    pattern_names: set[str] = set()
    pattern_predictors: list[tuple[str, ...]] = []
    resolved: list[str] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping) or set(raw_step) != {
            "pattern",
            "state_before_sha256",
            "state_after",
        }:
            raise ValueError(f"ACS target pattern step {index} changed: {path}.")
        pattern = raw_step.get("pattern")
        if not isinstance(pattern, str) or not pattern or pattern in pattern_names:
            raise ValueError(f"ACS target pattern name is invalid: {path}.")
        pattern_names.add(pattern)
        _require_sha256(
            raw_step.get("state_before_sha256"),
            label=f"ACS target predecessor state {path}",
        )
        state_payload = raw_step.get("state_after")
        if not isinstance(state_payload, Mapping):
            raise ValueError(f"ACS target state_after is malformed: {path}.")
        state = QRFChainState.from_dict(state_payload)
        if (
            state.entity != descriptor.get("entity")
            or state.targets != model_targets
            or not state.completed_targets
            or state.completed_targets[-1] != model_target
        ):
            raise ValueError(
                f"ACS target QRF state does not match its descriptor: {path}."
            )
        pattern_predictors.append(state.predictors)
        for predictor in state.predictors:
            if predictor not in resolved:
                resolved.append(predictor)
    return descriptor, tuple(pattern_predictors), tuple(resolved)


def _load_early_execution_evidence(
    artifact: Artifact,
    *,
    arm: str,
    family_specs: Sequence[FamilySpec],
    role_binding: Mapping[str, object] | None = None,
) -> EarlyExecutionEvidence:
    """Load and authenticate the operational early-transfer target bank."""

    binding = _validate_artifact_role(
        artifact,
        arm=arm,
        role_binding=role_binding,
    )
    early_specs = tuple(spec for spec in family_specs if spec.stage == "early_gap_fill")
    expected: dict[tuple[str, str, str, str], FamilySpec] = {}
    for spec in early_specs:
        direction = _early_direction(spec)
        targets = spec.baseline_targets if arm == "baseline" else spec.candidate_targets
        for target in targets:
            key = (direction, spec.entity, spec.family, target)
            if key in expected:
                raise ValueError(f"Early Phase-P target is declared twice: {key}.")
            expected[key] = spec
    expected_directions = {key[0] for key in expected}

    receipts_path = artifact.operational_receipts_path
    if not receipts_path.is_file():
        raise FileNotFoundError(
            f"Phase-P {arm} operational receipts are missing: {receipts_path}."
        )
    payload = json.loads(receipts_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Operational receipts are not an object: {receipts_path}.")
    expected_receipt_keys = {
        "artifact_kind",
        "schema_version",
        "materializer_version",
        "stage",
        "identity_sha256",
        "checkpoint",
        "operational_stage_receipts",
    }
    if set(payload) != expected_receipt_keys:
        raise ValueError(f"Operational receipts keys changed: {receipts_path}.")
    receipt_config = _execution_surface_config().get("operational_receipts")
    if not isinstance(receipt_config, Mapping):  # pragma: no cover - fixed config
        raise TypeError("Operational receipt config must be an object.")
    if any(payload.get(key) != value for key, value in receipt_config.items()):
        raise ValueError(f"Operational receipt envelope changed: {receipts_path}.")

    external = _external_metadata(artifact)
    if payload.get("identity_sha256") != external.get("identity_sha256"):
        raise ValueError(
            "Operational receipts do not bind the selected simulated stage identity."
        )
    checkpoint_sha256 = _sha256_file(artifact.checkpoint_path)
    checkpoint_receipt = payload.get("checkpoint")
    expected_checkpoint_receipt = {
        "filename": artifact.checkpoint_path.name,
        "sha256": checkpoint_sha256,
        "size_bytes": artifact.checkpoint_path.stat().st_size,
    }
    if checkpoint_receipt != expected_checkpoint_receipt:
        raise ValueError(
            "Operational receipts do not bind the selected simulated checkpoint."
        )

    target_bank = _nested(
        payload,
        "operational_stage_receipts",
        "impute",
        "acs_qrf_transfer",
        "target_bank",
    )
    if not isinstance(target_bank, Mapping) or set(target_bank) != {
        "directions",
        "identity_routing",
        "late_producer_groups",
    }:
        raise ValueError("Operational target-bank receipt shape changed.")
    directions = target_bank.get("directions")
    if not isinstance(directions, Mapping) or set(directions) != expected_directions:
        raise ValueError(
            "Operational early target-bank directions differ from the Phase-P specs."
        )
    configured_bank_ids = binding.get("early_target_bank_identity_sha256")
    if not isinstance(configured_bank_ids, Mapping) or set(configured_bank_ids) != (
        expected_directions
    ):
        raise ValueError(f"Phase-P {arm} bank identity configuration is incomplete.")

    executions: dict[tuple[str, str], EarlyTargetExecution] = {}
    for direction in sorted(expected_directions):
        bank = directions[direction]
        if not isinstance(bank, Mapping) or set(bank) != {
            "artifact_kind",
            "schema_version",
            "materializer_version",
            "root",
            "identity",
            "identity_sha256",
            "targets",
        }:
            raise ValueError(f"ACS target-bank receipt changed for {direction!r}.")
        expected_bank_sha256 = _require_sha256(
            configured_bank_ids[direction],
            label=f"Phase-P {arm} {direction} configured target bank",
        )
        bank_identity = bank.get("identity")
        if not isinstance(bank_identity, Mapping):
            raise ValueError(
                f"ACS target-bank identity is malformed for {direction!r}."
            )
        if (
            bank.get("artifact_kind") != ACS_TRANSFER_TARGET_BANK_RECEIPT_ARTIFACT_KIND
            or bank.get("schema_version") != ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION
            or bank.get("materializer_version")
            != ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION
            or bank.get("identity_sha256") != expected_bank_sha256
            or _bank_mapping_sha256(bank_identity) != expected_bank_sha256
        ):
            raise ValueError(
                f"ACS target-bank identity changed for Phase-P {arm} {direction}."
            )
        bank_root_raw = bank.get("root")
        if not isinstance(bank_root_raw, str) or not bank_root_raw:
            raise ValueError(f"ACS target-bank root is malformed for {direction!r}.")
        bank_root = Path(bank_root_raw).expanduser().resolve()
        checkpoint_identity_dir = artifact.checkpoint_path.parent.resolve()
        if not bank_root.is_relative_to(checkpoint_identity_dir):
            raise ValueError(
                f"ACS target-bank root escapes the selected checkpoint identity: "
                f"{bank_root}."
            )
        targets_receipt = bank.get("targets")
        if not isinstance(targets_receipt, Mapping) or not targets_receipt:
            raise ValueError(
                f"ACS target-bank targets are malformed for {direction!r}."
            )
        try:
            indexes = sorted(int(index) for index in targets_receipt)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"ACS target-bank indexes are malformed for {direction!r}."
            ) from error
        if indexes != list(range(len(indexes))):
            raise ValueError(
                f"ACS target-bank indexes are not contiguous: {direction}."
            )
        for target_index in indexes:
            record = targets_receipt[str(target_index)]
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"ACS target receipt {direction}/{target_index} changed."
                )
            required_record_keys = {
                "source",
                "descriptor",
                "load_status",
                "path",
                "checkpoint_sha256",
                "size_bytes",
                "raw_draw_sha256",
                "content_metadata_sha256",
                "write_status",
                "write_seconds",
            }
            if (
                set(record) != required_record_keys
                or record.get("source") not in {"rebuilt", "checkpoint"}
                or record.get("load_status") not in {"loaded", "missing"}
                or record.get("write_status") not in {"rebuilt", "not_required"}
                or isinstance(record.get("write_seconds"), bool)
                or not isinstance(record.get("write_seconds"), (int, float))
                or not math.isfinite(float(record["write_seconds"]))
                or float(record["write_seconds"]) < 0.0
            ):
                raise ValueError(
                    f"ACS target receipt {direction}/{target_index} changed."
                )
            target_path_raw = record.get("path")
            if not isinstance(target_path_raw, str) or not target_path_raw:
                raise ValueError(f"ACS target path {direction}/{target_index} changed.")
            target_path = Path(target_path_raw).expanduser().resolve()
            if not target_path.is_relative_to(bank_root / "targets"):
                raise ValueError(
                    f"ACS target path escapes its bank root: {target_path}."
                )
            descriptor, pattern_predictors, resolved = _read_target_execution(
                target_path,
                record=record,
                bank_identity=bank_identity,
                bank_identity_sha256=expected_bank_sha256,
            )
            if descriptor.get("target_index") != target_index or descriptor.get(
                "total_targets"
            ) != len(indexes):
                raise ValueError(
                    f"ACS target index binding changed for {direction}/{target_index}."
                )
            entity = descriptor.get("entity")
            family_raw = descriptor.get("family")
            if not isinstance(entity, str) or not isinstance(family_raw, str):
                raise ValueError(f"ACS target owner is malformed: {target_path}.")
            family = _BATCH_SUFFIX.sub("", family_raw)
            exported_targets = _string_list(
                descriptor.get("exported_targets"),
                label=f"ACS exported targets {target_path}",
            )
            for target in exported_targets:
                expected_key = (direction, entity, family, target)
                spec = expected.get(expected_key)
                if spec is None:
                    raise ValueError(
                        f"ACS target-bank exported undeclared Phase-P target {expected_key}."
                    )
                execution_key = (spec.family_id, target)
                if execution_key in executions:
                    raise ValueError(
                        f"ACS target-bank repeats Phase-P target {execution_key}."
                    )
                executions[execution_key] = EarlyTargetExecution(
                    family_id=spec.family_id,
                    direction=direction,
                    entity=entity,
                    family=family,
                    target=target,
                    bank_identity_sha256=expected_bank_sha256,
                    checkpoint_path=target_path,
                    checkpoint_sha256=str(record["checkpoint_sha256"]),
                    pattern_predictors=pattern_predictors,
                    resolved_predictors=resolved,
                )

    expected_execution_keys = {
        (spec.family_id, target)
        for spec in early_specs
        for target in (
            spec.baseline_targets if arm == "baseline" else spec.candidate_targets
        )
    }
    if set(executions) != expected_execution_keys:
        raise ValueError(
            f"Phase-P {arm} target-bank coverage changed: "
            f"missing={sorted(expected_execution_keys - set(executions))}, "
            f"extra={sorted(set(executions) - expected_execution_keys)}."
        )
    _validate_early_predictor_evidence(
        arm=arm,
        family_specs=early_specs,
        executions=executions,
    )
    return EarlyExecutionEvidence(
        receipts_path=receipts_path,
        receipts_sha256=_sha256_file(receipts_path),
        simulated_checkpoint_sha256=checkpoint_sha256,
        targets=executions,
    )


def _early_investment_config() -> Mapping[str, object]:
    value = _execution_surface_config().get("early_investment")
    if not isinstance(value, Mapping):  # pragma: no cover - fixed config
        raise TypeError("Phase-P early investment config must be an object.")
    return value


def _entity_investment_feature(entity: str) -> tuple[str, str]:
    config = _early_investment_config()
    person_feature = str(config["person_feature"])
    group_feature = str(config["group_feature"])
    if entity == "person":
        return person_feature, group_feature
    if entity == "spm_unit":
        return group_feature, person_feature
    raise ValueError(f"Unsupported early investment entity {entity!r}.")


def _validate_early_predictor_evidence(
    *,
    arm: str,
    family_specs: Sequence[FamilySpec],
    executions: Mapping[tuple[str, str], EarlyTargetExecution],
) -> None:
    """Bind declared predictors to pattern states and their target-level union.

    Required predictors must occur in every source-pattern state. Optional
    predictors are pattern-specific, so exact equality applies to their union
    across states. The baseline early investment feature is the sole declared
    exception: it was unavailable before post-PUF leaves existed.
    """

    for spec in family_specs:
        feature, opposite_feature = _entity_investment_feature(spec.entity)
        predictor_set = (
            spec.baseline_predictor_set
            if arm == "baseline"
            else spec.candidate_predictor_set
        )
        if feature in predictor_set.required:
            raise ValueError(
                f"Early investment predictor became required for {spec.family_id}."
            )
        declared = feature in predictor_set.optional
        expected_present = arm == "candidate" and declared
        required = set(predictor_set.required)
        allowed = required.union(predictor_set.optional)
        targets = spec.baseline_targets if arm == "baseline" else spec.candidate_targets
        for target in targets:
            execution = executions[(spec.family_id, target)]
            observed = set(execution.resolved_predictors)
            if opposite_feature in observed:
                raise ValueError(
                    f"Early target {spec.family_id}/{target} resolved the "
                    "opposite-grain investment predictor."
                )
            if (feature in observed) != expected_present:
                raise ValueError(
                    f"Phase-P {arm} early target {spec.family_id}/{target} "
                    f"investment evidence changed: declared={declared}, "
                    f"observed={feature in observed}, expected={expected_present}."
                )
            for pattern_index, pattern_predictors in enumerate(
                execution.pattern_predictors
            ):
                pattern = set(pattern_predictors)
                missing_required = required - pattern
                undeclared = pattern - allowed
                if missing_required or undeclared:
                    raise ValueError(
                        f"Phase-P {arm} early target {spec.family_id}/{target} "
                        f"pattern {pattern_index} predictor surface changed: "
                        f"missing_required={sorted(missing_required)}, "
                        f"undeclared={sorted(undeclared)}."
                    )
            expected_union = set(allowed)
            if arm == "baseline":
                expected_union.discard(feature)
            if observed != expected_union:
                raise ValueError(
                    f"Phase-P {arm} early target {spec.family_id}/{target} "
                    "resolved predictor union changed: "
                    f"missing={sorted(expected_union - observed)}, "
                    f"extra={sorted(observed - expected_union)}."
                )


def _artifact_target_orders(artifact: Artifact) -> dict[str, tuple[str, ...]]:
    external = _external_metadata(artifact)
    impute = _nested(external, "stage_receipts", "impute")
    if not isinstance(impute, Mapping):
        raise ValueError("Artifact impute stage receipt is not an object.")
    families = _nested(impute, "acs_qrf_transfer", "target_families")
    if not isinstance(families, Mapping):
        raise ValueError("Artifact target_families receipt is not an object.")

    result: dict[str, tuple[str, ...]] = {}
    early = families.get("early_gap_fill")
    if not isinstance(early, Mapping):
        raise ValueError("Artifact target_families lacks early_gap_fill.")
    for entity, entity_families in early.items():
        if not isinstance(entity_families, Mapping):
            raise ValueError(f"Early target families for {entity!r} are malformed.")
        for family, targets in entity_families.items():
            if not isinstance(targets, list) or not all(
                isinstance(target, str) for target in targets
            ):
                raise ValueError(
                    f"Early target order for {entity}/{family} is malformed."
                )
            result[f"early_gap_fill/{entity}/{family}"] = tuple(targets)

    late = _nested(impute, "stacked_post_puf_transfer", "groups")
    if not isinstance(late, Mapping):
        raise ValueError("Artifact late-transfer groups receipt is not an object.")
    for group_name, receipt in late.items():
        if not isinstance(receipt, Mapping):
            raise ValueError(f"Late group receipt {group_name!r} is malformed.")
        targets = receipt.get("ordered_targets")
        if not isinstance(targets, list) or not all(
            isinstance(target, str) for target in targets
        ):
            raise ValueError(f"Late target order for {group_name!r} is malformed.")
        result[f"late_transfer/{group_name}"] = tuple(targets)
    return result


def _load_lineage_spec() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "specs" / "us_imputation_lineage.yaml"
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Lineage spec {path} is not a mapping.")
    return value


def production_family_specs(
    baseline: Artifact,
    candidate: Artifact,
) -> tuple[FamilySpec, ...]:
    """Bind the 32 changed families to both artifacts' executed target order."""

    spec = _load_lineage_spec()
    declared = spec.get("imputed_families")
    if not isinstance(declared, list) or not declared:
        raise ValueError("Lineage spec imputed_families must be a nonempty list.")
    baseline_orders = _artifact_target_orders(baseline)
    candidate_orders = _artifact_target_orders(candidate)

    result: list[FamilySpec] = []
    observed_ids: set[str] = set()
    for raw in declared:
        if not isinstance(raw, Mapping):
            raise ValueError("Lineage imputed family entry is not an object.")
        family_id = str(raw["id"])
        stage = str(raw["stage"])
        entity = str(raw["entity"])
        family = str(raw["family"])
        targets = tuple(str(item) for item in raw["targets"])
        # Early families declare their runtime bound explicitly.  Each late
        # family is already one registry-bounded transfer group, so its full
        # declared target tuple is the chain.
        max_targets = int(raw.get("max_targets_per_fit", len(targets)))
        if family_id in observed_ids:
            raise ValueError(f"Lineage spec repeats family id {family_id!r}.")
        observed_ids.add(family_id)

        if stage == "early_gap_fill":
            order_key = f"early_gap_fill/{entity}/{family}"
        elif stage == "late_transfer":
            order_key = family_id
        else:
            raise ValueError(f"Unsupported Phase-P family stage {stage!r}.")
        try:
            baseline_targets = baseline_orders[order_key]
            candidate_targets = candidate_orders[order_key]
        except KeyError as exc:
            raise ValueError(
                f"Artifact target receipts do not bind family {family_id!r}."
            ) from exc
        if set(baseline_targets) != set(targets):
            raise ValueError(
                f"Baseline targets for {family_id!r} differ from the Phase-P "
                f"declaration: receipt={baseline_targets}, declared={targets}."
            )
        if candidate_targets != targets:
            raise ValueError(
                f"Candidate target order for {family_id!r} differs from the "
                f"Phase-P declaration: receipt={candidate_targets}, "
                f"declared={targets}."
            )

        baseline_predictors = _legacy_predictor_set(
            entity=entity,
            targets=baseline_targets,
        )
        surface = {entity: {family: candidate_targets}}
        if stage == "early_gap_fill":
            candidate_predictors = declared_acs_transfer_predictor_sets(surface)[
                entity
            ][family]
        else:
            candidate_predictors = declared_stacked_post_puf_predictor_sets(surface)[
                entity
            ][family]
        result.append(
            FamilySpec(
                family_id=family_id,
                stage=stage,
                entity=entity,
                family=family,
                baseline_targets=baseline_targets,
                candidate_targets=candidate_targets,
                max_targets_per_fit=max_targets,
                baseline_predictor_set=baseline_predictors,
                candidate_predictor_set=candidate_predictors,
            )
        )
    if len(result) != 32:
        raise ValueError(
            f"Phase-P lineage must bind exactly 32 changed families; got {len(result)}."
        )
    return tuple(result)


def _donor_projection(frame: Frame, *, clone_index: int) -> Frame:
    person = frame.table(frame.schema.person_entity)
    required = {"person_support_channel", "person_support_clone_index"}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(f"OOS donor projection lacks person columns {missing}.")
    mask = (
        person["person_support_channel"].astype(str).eq(_ASEC_CHANNEL)
        & pd.to_numeric(person["person_support_clone_index"], errors="raise").eq(
            clone_index
        )
    ).to_numpy()
    if not mask.any():
        raise ValueError(
            f"OOS donor projection found no ASEC clone-{clone_index} person rows."
        )
    return frame.select(mask)


def _source_ids(frame: Frame, *, entity: str) -> np.ndarray:
    table = frame.table(entity)
    column = f"{entity}_source_id"
    if column not in table:
        raise ValueError(f"OOS donor entity {entity!r} lacks stable id {column!r}.")
    values = table[column].astype(str).to_numpy(dtype=object)
    if pd.isna(table[column]).any():
        raise ValueError(f"OOS donor stable id {column!r} contains nulls.")
    if len(set(values.tolist())) != len(values):
        raise ValueError(f"OOS donor stable id {column!r} is not unique.")
    return values


def _folds_and_clusters(
    ids: np.ndarray, *, entity: str
) -> tuple[np.ndarray, np.ndarray]:
    fold = PHASE_P_CONFIG["fold"]
    bootstrap = PHASE_P_CONFIG["bootstrap"]
    assert isinstance(fold, Mapping) and isinstance(bootstrap, Mapping)
    fold_namespace = f"phase-p-fold-v1:{int(fold['seed'])}:{entity}"
    cluster_namespace = f"phase-p-bootstrap-v1:{int(bootstrap['seed'])}:{entity}"
    folds = np.fromiter(
        (
            _stable_bucket(
                item,
                namespace=fold_namespace,
                buckets=int(fold["fold_count"]),
            )
            for item in ids
        ),
        dtype=np.int16,
        count=len(ids),
    )
    clusters = np.fromiter(
        (
            _stable_bucket(
                item,
                namespace=cluster_namespace,
                buckets=int(bootstrap["clusters"]),
            )
            for item in ids
        ),
        dtype=np.int16,
        count=len(ids),
    )
    return folds, clusters


def _as_float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            for column in frame.columns
        },
        index=frame.index,
    )


def _prepare_design(
    features: pd.DataFrame,
    *,
    required: Sequence[str],
    optional: Sequence[str],
    predecessors: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return explicit train/test matrices and masks for one target."""

    fold = PHASE_P_CONFIG["fold"]
    assert isinstance(fold, Mapping)
    held_out = int(fold["held_out_fold"])
    base = _as_float_frame(features.loc[:, [*required, *optional]])
    chain = _as_float_frame(predecessors)
    required_matrix = pd.concat([base.loc[:, list(required)], chain], axis=1)
    required_complete = np.isfinite(required_matrix.to_numpy(copy=False)).all(axis=1)
    valid = required_complete & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    train_mask = valid & (folds != held_out)
    test_mask = valid & (folds == held_out)

    train_parts = [required_matrix.loc[train_mask].to_numpy(dtype=np.float64)]
    test_parts = [required_matrix.loc[test_mask].to_numpy(dtype=np.float64)]
    unavailable: list[str] = []
    for predictor in optional:
        train_values = base.loc[train_mask, predictor].to_numpy(dtype=np.float64)
        test_values = base.loc[test_mask, predictor].to_numpy(dtype=np.float64)
        finite = np.isfinite(train_values)
        if not finite.any():
            unavailable.append(predictor)
            continue
        median = float(np.median(train_values[finite]))
        train_parts.extend(
            [
                np.where(finite, train_values, median)[:, None],
                finite.astype(np.float64)[:, None],
            ]
        )
        test_finite = np.isfinite(test_values)
        test_parts.extend(
            [
                np.where(test_finite, test_values, median)[:, None],
                test_finite.astype(np.float64)[:, None],
            ]
        )
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        y[train_mask],
        y[test_mask],
        weights[train_mask],
        tuple(unavailable),
    )


def _fit_target_loss(
    *,
    key: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    train_weights: np.ndarray,
) -> np.ndarray:
    forest = PHASE_P_CONFIG["forest"]
    loss_config = PHASE_P_CONFIG["quantile_loss"]
    quantiles = np.asarray(PHASE_P_CONFIG["quantiles"], dtype=np.float64)
    assert isinstance(forest, Mapping) and isinstance(loss_config, Mapping)
    if loss_config.get("per_row_quantile_aggregation") != (
        "unweighted_arithmetic_mean"
    ):
        raise ValueError("Unsupported Phase-P per-row quantile-loss aggregation.")
    minimum_train = int(forest["minimum_training_rows"])
    minimum_test = int(forest["minimum_holdout_rows"])
    if len(train_y) < minimum_train or len(test_y) < minimum_test:
        raise ValueError(
            f"OOS target {key} has insufficient donors: train={len(train_y)} "
            f"(minimum {minimum_train}), holdout={len(test_y)} "
            f"(minimum {minimum_test})."
        )
    probability = train_weights / train_weights.sum()
    rng = np.random.default_rng(_seed("phase-p-oos-fit", forest["seed"], key))
    sampled = rng.choice(len(train_y), size=len(train_y), replace=True, p=probability)
    model = RandomForestQuantileRegressor(
        n_estimators=int(forest["n_estimators"]),
        default_quantiles=float(forest["default_quantiles"]),
        criterion=str(forest["criterion"]),
        max_depth=int(forest["max_depth"]),
        min_samples_split=int(forest["min_samples_split"]),
        min_samples_leaf=int(forest["min_samples_leaf"]),
        max_samples_leaf=int(forest["max_samples_leaf"]),
        min_weight_fraction_leaf=float(forest["min_weight_fraction_leaf"]),
        max_features=float(forest["max_features"]),
        max_leaf_nodes=forest["max_leaf_nodes"],
        min_impurity_decrease=float(forest["min_impurity_decrease"]),
        bootstrap=bool(forest["bootstrap"]),
        oob_score=bool(forest["oob_score"]),
        n_jobs=int(forest["n_jobs"]),
        random_state=_seed("phase-p-oos-forest", forest["seed"], key),
        verbose=int(forest["verbose"]),
        warm_start=bool(forest["warm_start"]),
        ccp_alpha=float(forest["ccp_alpha"]),
        max_samples=float(forest["max_samples"]),
        monotonic_cst=forest["monotonic_cst"],
    )
    model.fit(train_x[sampled], train_y[sampled])
    predicted = model.predict(
        test_x,
        quantiles=quantiles.tolist(),
        interpolation=str(forest["predict_interpolation"]),
        weighted_quantile=bool(forest["predict_weighted_quantile"]),
        weighted_leaves=bool(forest["predict_weighted_leaves"]),
        aggregate_leaves_first=bool(forest["predict_aggregate_leaves_first"]),
        oob_score=False,
        indices=None,
        duplicates=None,
    )
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != (len(test_y), len(quantiles)):
        raise RuntimeError(
            f"OOS target {key} produced quantile matrix {predicted.shape}; "
            f"expected {(len(test_y), len(quantiles))}."
        )
    residual = test_y[:, None] - predicted
    losses = np.maximum(quantiles * residual, (quantiles - 1.0) * residual)
    return losses.mean(axis=1)


def _bounded_chunks(targets: Sequence[str], width: int) -> list[tuple[str, ...]]:
    return [
        tuple(targets[start : start + width]) for start in range(0, len(targets), width)
    ]


def _phase_p_oos_feature_surface(
    donor: Frame,
    *,
    spec: FamilySpec,
    arm: str,
    predictor_set: AcsPredictorSet,
    execution_evidence: EarlyExecutionEvidence | None,
) -> OOSFeatureSurface:
    """Reconstruct the predictor surface at its actual execution stage."""

    targets = spec.baseline_targets if arm == "baseline" else spec.candidate_targets
    if spec.stage != "early_gap_fill":
        surface = _transfer_feature_surface(
            donor,
            donor,
            entity=spec.entity,
            targets=targets,
            predictor_set=predictor_set,
            require_all_predictors=False,
        )
        return OOSFeatureSurface(
            features=surface.donor,
            required=tuple(surface.required),
            optional=tuple(surface.optional),
            source_bindings={
                "execution_surface": "final_simulated_checkpoint",
                "stage": spec.stage,
            },
        )
    if execution_evidence is None:
        raise ValueError(
            f"Early Phase-P family {spec.family_id} lacks execution evidence."
        )

    feature, _opposite_feature = _entity_investment_feature(spec.entity)
    _validate_early_predictor_evidence(
        arm=arm,
        family_specs=(spec,),
        executions=execution_evidence.targets,
    )
    if feature in predictor_set.required:
        raise ValueError(
            f"Early investment predictor became required for {spec.family_id}."
        )
    declared = feature in predictor_set.optional
    stripped_predictors = AcsPredictorSet(
        required=predictor_set.required,
        optional=tuple(name for name in predictor_set.optional if name != feature),
    )
    surface = _transfer_feature_surface(
        donor,
        donor,
        entity=spec.entity,
        targets=targets,
        predictor_set=stripped_predictors,
        require_all_predictors=False,
    )
    features = surface.donor.copy()
    source_mode = "not_declared"
    config = _early_investment_config()
    components: tuple[str, ...] = ()
    if arm == "baseline" and declared:
        source_mode = str(config["baseline_donor_source"])
    elif arm == "candidate" and declared:
        raw_components = config.get("candidate_donor_components")
        if not isinstance(raw_components, list) or not all(
            isinstance(component, str) and component for component in raw_components
        ):
            raise TypeError("Candidate donor component config is malformed.")
        components = tuple(raw_components)
        person_feature = str(config["person_feature"])
        person_values = _complete_numeric_component_sum(
            donor,
            components,
            feature=person_feature,
        )
        if person_values is None:
            raise ValueError(
                f"Candidate early donor lacks raw investment components {components}."
            )
        if spec.entity == "person":
            feature_values = person_values
            source_mode = str(config["candidate_person_donor_source"])
        else:
            grouped = _aggregate_optional_to_group(
                donor,
                entity=spec.entity,
                person_features=person_values.to_frame(),
            )
            feature_values = grouped[feature]
            source_mode = "complete_numeric_component_sum_then_group_projection"
        features[feature] = feature_values.to_numpy()

    resolved_optional = tuple(
        name for name in predictor_set.optional if name in features.columns
    )
    features = features.loc[:, [*surface.required, *resolved_optional]]
    target_receipts = {
        target: {
            "direction": execution.direction,
            "bank_identity_sha256": execution.bank_identity_sha256,
            "checkpoint_path": str(execution.checkpoint_path),
            "checkpoint_sha256": execution.checkpoint_sha256,
            "resolved_predictors": list(execution.resolved_predictors),
            "pattern_predictors": [
                list(predictors) for predictors in execution.pattern_predictors
            ],
            "investment_predictor_observed": (feature in execution.resolved_predictors),
        }
        for target in targets
        for execution in [execution_evidence.targets[(spec.family_id, target)]]
    }
    return OOSFeatureSurface(
        features=features,
        required=tuple(surface.required),
        optional=resolved_optional,
        source_bindings={
            "execution_surface": "early_gap_fill_target_bank_replay",
            "operational_receipts_path": str(execution_evidence.receipts_path),
            "operational_receipts_sha256": execution_evidence.receipts_sha256,
            "simulated_checkpoint_sha256": (
                execution_evidence.simulated_checkpoint_sha256
            ),
            "investment_income": {
                "feature": feature,
                "declared_optional": declared,
                "oos_donor_source": source_mode,
                "raw_components": list(components),
                "group_projection": (
                    config["candidate_group_projection"]
                    if arm == "candidate" and declared and spec.entity != "person"
                    else None
                ),
            },
            "target_checkpoints": target_receipts,
        },
    )


def _measure_family(
    donor: Frame,
    *,
    spec: FamilySpec,
    arm: str,
    execution_evidence: EarlyExecutionEvidence | None,
) -> dict[str, OOSTargetMeasurement]:
    targets = spec.baseline_targets if arm == "baseline" else spec.candidate_targets
    predictor_set = (
        spec.baseline_predictor_set
        if arm == "baseline"
        else spec.candidate_predictor_set
    )
    surface = _phase_p_oos_feature_surface(
        donor,
        spec=spec,
        arm=arm,
        predictor_set=predictor_set,
        execution_evidence=execution_evidence,
    )
    resolved_optional = tuple(surface.optional)
    missing_optional = tuple(
        predictor
        for predictor in predictor_set.optional
        if predictor not in resolved_optional
    )
    table = donor.table(spec.entity)
    ids = _source_ids(donor, entity=spec.entity)
    weights = donor.resolve_weights(spec.entity).values
    folds, clusters = _folds_and_clusters(ids, entity=spec.entity)
    holdout_mask = folds == int(PHASE_P_CONFIG["fold"]["held_out_fold"])
    results: dict[str, OOSTargetMeasurement] = {}

    for chunk in _bounded_chunks(targets, spec.max_targets_per_fit):
        complete = _complete_target_mask(table, targets=chunk)
        encodings = _complete_case_target_encodings(
            table,
            targets=chunk,
            complete=complete,
        )
        model_order: list[str] = []
        model_values: dict[str, pd.Series] = {}
        targets_by_model: dict[str, list[str]] = {}
        for target in chunk:
            encoding = encodings[target]
            targets_by_model.setdefault(encoding.model_target, []).append(target)
            if encoding.model_target not in model_values:
                model_order.append(encoding.model_target)
                model_values[encoding.model_target] = encoding.model_values

        for model_position, model_target in enumerate(model_order):
            exported_targets = targets_by_model[model_target]
            predecessors = tuple(model_order[:model_position])
            predecessor_frame = pd.DataFrame(
                {
                    f"__phase_p_chain_{position:02d}": model_values[name]
                    for position, name in enumerate(predecessors)
                },
                index=table.index,
            )
            y = model_values[model_target].to_numpy(dtype=np.float64)
            (
                train_x,
                test_x,
                train_y,
                test_y,
                train_weights,
                all_missing_optional,
            ) = _prepare_design(
                surface.features,
                required=surface.required,
                optional=resolved_optional,
                predecessors=predecessor_frame,
                y=y,
                weights=weights,
                folds=folds,
            )
            # Excluding the arm from the seed key is deliberate: paired arms
            # consume the same bootstrap/forest randomness for a target, so
            # the loss delta is not polluted by independent Monte Carlo noise.
            model_key = f"{spec.family_id}/{model_target}"
            losses = _fit_target_loss(
                key=model_key,
                train_x=train_x,
                test_x=test_x,
                train_y=train_y,
                test_y=test_y,
                train_weights=train_weights,
            )

            valid = (
                np.isfinite(y)
                & np.isfinite(weights)
                & (weights > 0)
                & np.isfinite(
                    _as_float_frame(
                        pd.concat(
                            [
                                surface.features.loc[:, list(surface.required)],
                                predecessor_frame,
                            ],
                            axis=1,
                        )
                    ).to_numpy(copy=False)
                ).all(axis=1)
            )
            test_positions = np.flatnonzero(valid & holdout_mask)
            if len(test_positions) != len(losses):
                raise RuntimeError(
                    f"OOS target {model_key} lost holdout row alignment."
                )
            unavailable = tuple(
                dict.fromkeys([*missing_optional, *all_missing_optional])
            )
            resolved = tuple(
                [
                    *surface.required,
                    *(
                        predictor
                        for predictor in resolved_optional
                        if predictor not in all_missing_optional
                    ),
                    *predecessors,
                ]
            )
            for target in exported_targets:
                key = f"{spec.family_id}/{target}"
                results[key] = OOSTargetMeasurement(
                    key=key,
                    family_id=spec.family_id,
                    stage=spec.stage,
                    entity=spec.entity,
                    family=spec.family,
                    target=target,
                    target_position=targets.index(target),
                    chain_predecessors=predecessors,
                    predictor_set={
                        "required": list(predictor_set.required),
                        "optional": list(predictor_set.optional),
                    },
                    resolved_predictors=resolved,
                    unavailable_optional_predictors=unavailable,
                    training_rows=len(train_y),
                    holdout_rows=len(test_y),
                    ids=ids[test_positions].copy(),
                    weights=weights[test_positions].copy(),
                    truth=test_y.copy(),
                    losses=losses.copy(),
                    clusters=clusters[test_positions].copy(),
                    predictor_source_bindings=surface.source_bindings,
                )
    expected = {f"{spec.family_id}/{target}" for target in targets}
    if set(results) != expected:
        raise RuntimeError(
            f"OOS family {spec.family_id!r} did not measure every target: "
            f"missing={sorted(expected - set(results))}, "
            f"extra={sorted(set(results) - expected)}."
        )
    return results


def _champva_incidence(frame: Frame) -> dict[str, IncidenceMeasurement]:
    person = frame.table("person")
    target = "has_champva_health_coverage_at_interview"
    required = {
        target,
        "person_source_id",
        "person_support_channel",
        "person_support_clone_index",
    }
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(f"CHAMPVA bootstrap lacks person columns {missing}.")
    weights = frame.resolve_weights("person").values
    channel = person["person_support_channel"].astype(str)
    clone = pd.to_numeric(person["person_support_clone_index"], errors="raise")
    values = person[target].to_numpy(dtype=np.float64, na_value=np.nan)
    results: dict[str, IncidenceMeasurement] = {}
    for origin in (_ASEC_CHANNEL, _ACS_CHANNEL):
        mask = (
            channel.eq(origin).to_numpy()
            & clone.eq(_CLONE_ZERO).to_numpy()
            & np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0)
        )
        ids = person.loc[mask, "person_source_id"].astype(str).to_numpy(dtype=object)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"CHAMPVA {origin} source ids are not unique.")
        _, clusters = _folds_and_clusters(ids, entity=f"person:{origin}")
        results[origin] = IncidenceMeasurement(
            origin=origin,
            ids=ids,
            weights=weights[mask].copy(),
            values=values[mask].copy(),
            clusters=clusters,
        )
    return results


def measure_arm(
    artifact: Artifact,
    *,
    arm: str,
    family_specs: Sequence[FamilySpec],
    execution_evidence: EarlyExecutionEvidence,
) -> ArmMeasurement:
    """Load one large checkpoint once, retain only compact OOS measurements."""

    loaded = load_frame_checkpoint(artifact.checkpoint_path)
    frame = loaded.frame
    incidence = _champva_incidence(frame)
    targets: dict[str, OOSTargetMeasurement] = {}
    completed = 0
    for clone_index in (_CLONE_ZERO, _PUF_CLONE):
        matching_specs = [
            spec for spec in family_specs if spec.donor_clone_index == clone_index
        ]
        if not matching_specs:
            continue
        donor = _donor_projection(frame, clone_index=clone_index)
        for spec in matching_specs:
            completed += 1
            print(
                f"phase-p-oos: {arm} family {completed}/{len(family_specs)} "
                f"{spec.family_id}",
                flush=True,
            )
            try:
                measured = _measure_family(
                    donor,
                    spec=spec,
                    arm=arm,
                    execution_evidence=execution_evidence,
                )
            except Exception as error:
                raise GateMeasurementError(
                    arm=arm,
                    family_id=spec.family_id,
                    stage=spec.stage,
                    entity=spec.entity,
                    family=spec.family,
                    cause=error,
                ) from error
            overlap = sorted(set(targets).intersection(measured))
            if overlap:
                raise RuntimeError(
                    f"OOS target keys repeat across families: {overlap}."
                )
            targets.update(measured)
        del donor
        gc.collect()
    del frame, loaded
    gc.collect()
    return ArmMeasurement(targets=targets, champva_by_origin=incidence)


def _battery_receipt_config() -> Mapping[str, object]:
    value = PHASE_P_CONFIG.get("battery_receipt")
    if not isinstance(value, Mapping):  # pragma: no cover - fixed config
        raise TypeError("Phase-P battery_receipt config must be an object.")
    return value


def _battery_gate(artifact: Artifact) -> dict[str, object]:
    """Validate and return the production battery receipt for one artifact."""

    config = _battery_receipt_config()
    expected_top_level_keys = set(
        _string_list(
            config.get("top_level_keys"),
            label="Phase-P battery top-level keys",
        )
    )
    if set(artifact.gates) != expected_top_level_keys:
        raise ValueError(
            f"Battery diagnostics envelope keys changed in {artifact.gates_path}."
        )
    for key in ("artifact_kind", "schema_version", "semantic_kind", "pipeline"):
        if artifact.gates.get(key) != config.get(key):
            raise ValueError(
                f"Battery diagnostics envelope field {key!r} changed in "
                f"{artifact.gates_path}."
            )
    for key in ("publication_run_id", "release_id"):
        value = artifact.gates.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Battery diagnostics envelope field {key!r} is malformed in "
                f"{artifact.gates_path}."
            )
    if not isinstance(artifact.gates.get("simulation_ready"), bool):
        raise ValueError(
            f"Battery diagnostics simulation_ready is malformed in "
            f"{artifact.gates_path}."
        )

    agreement = artifact.gates.get("agreement_gate")
    terminal = artifact.gates.get("terminal_gates")
    if (
        not isinstance(agreement, Mapping)
        or not isinstance(terminal, Mapping)
        or agreement != terminal
        or set(terminal) != {"gates", "passed"}
        or not isinstance(terminal.get("passed"), bool)
    ):
        raise ValueError(
            f"Battery agreement/terminal gate envelopes diverged in "
            f"{artifact.gates_path}."
        )
    gates = terminal.get("gates")
    gate_id = str(config["gate_id"])
    if not isinstance(gates, Mapping) or gate_id not in gates:
        raise ValueError(f"Battery gate is missing in {artifact.gates_path}.")
    value = gates[gate_id]
    if not isinstance(value, dict) or set(value) != set(
        _string_list(config.get("gate_keys"), label="Phase-P battery gate keys")
    ):
        raise ValueError(f"Battery gate in {artifact.gates_path} is malformed.")

    failures = value.get("failures")
    if not isinstance(failures, list) or not all(
        isinstance(failure, str) for failure in failures
    ):
        raise ValueError(f"Battery failures in {artifact.gates_path} are malformed.")
    if not isinstance(value.get("passed"), bool) or value["passed"] != (not failures):
        raise ValueError(
            f"Battery passed/failures receipt is inconsistent in {artifact.gates_path}."
        )

    details = value.get("details")
    if not isinstance(details, Mapping) or set(details) != set(
        _string_list(
            config.get("details_keys"),
            label="Phase-P battery detail keys",
        )
    ):
        raise ValueError(f"Battery details in {artifact.gates_path} are malformed.")
    checkpoint_authority = _nested(
        _external_metadata(artifact),
        "identity",
        "stacked_authority",
    )
    authority = details.get("authority")
    if not isinstance(authority, Mapping) or authority != checkpoint_authority:
        raise ValueError(
            f"Battery authority does not match the selected checkpoint authority in "
            f"{artifact.gates_path}."
        )
    components = authority.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("Battery authority components are malformed.")
    declared_surface = components.get("declared_surface")
    gap_fill_plan = components.get("gap_fill_plan")
    joint_registry = components.get("joint_metric_registry")
    support_profile = components.get("support_profile")
    if not all(
        isinstance(component, Mapping)
        for component in (
            declared_surface,
            gap_fill_plan,
            joint_registry,
            support_profile,
        )
    ):
        raise ValueError("Battery authority lacks required component receipts.")

    declared_target_count = declared_surface.get("target_count")
    joint_target_count = joint_registry.get("target_count")
    if (
        isinstance(declared_target_count, bool)
        or not isinstance(declared_target_count, int)
        or declared_target_count <= 0
        or isinstance(joint_target_count, bool)
        or not isinstance(joint_target_count, int)
        or joint_target_count < 0
    ):
        raise ValueError("Battery authority target counts are malformed.")
    expected_comparison_count = declared_target_count + joint_target_count
    comparisons = details.get("comparisons")
    if not isinstance(comparisons, Mapping) or len(comparisons) != (
        expected_comparison_count
    ):
        raise ValueError(
            f"Battery comparison coverage changed in {artifact.gates_path}: "
            f"observed={len(comparisons) if isinstance(comparisons, Mapping) else None}, "
            f"expected={expected_comparison_count}."
        )
    registered_specs = details.get("registered_specs")
    if (
        details.get("declared_target_count") != declared_target_count
        or details.get("registered_target_count") != declared_target_count
        or details.get("registered_joint_target_count") != joint_target_count
        or details.get("tested_comparisons") != expected_comparison_count
        or isinstance(registered_specs, bool)
        or not isinstance(registered_specs, int)
        or registered_specs <= 0
        or details.get("missing_declared_targets") != []
        or details.get("extra_registered_targets") != []
        or details.get("untestable_comparisons") != []
    ):
        raise ValueError(
            f"Battery completeness receipt changed in {artifact.gates_path}."
        )
    expected_plan = {
        "plan_id": config["declared_plan_id"],
        "sha256": gap_fill_plan.get("sha256"),
        "version": authority.get("version"),
    }
    if details.get("declared_plan") != expected_plan:
        raise ValueError(
            f"Battery declared-plan binding changed in {artifact.gates_path}."
        )
    if details.get("support_profile") != support_profile:
        raise ValueError(
            f"Battery support-profile binding changed in {artifact.gates_path}."
        )
    if not isinstance(details.get("puf_capital_gains_tail_support"), Mapping):
        raise ValueError(
            f"Battery PUF tail support receipt is malformed in {artifact.gates_path}."
        )
    if details.get("tolerances") != config.get("tolerances"):
        raise ValueError(f"Battery tolerances changed in {artifact.gates_path}.")
    return value


def _battery_legs(artifact: Artifact) -> dict[str, dict[str, object]]:
    gate = _battery_gate(artifact)
    battery = gate["details"]
    assert isinstance(battery, Mapping)
    comparisons = battery.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError(f"Battery comparisons in {artifact.gates_path} are malformed.")
    failures = gate.get("failures")
    if not isinstance(failures, list) or not all(
        isinstance(failure, str) for failure in failures
    ):
        raise ValueError(f"Battery failures in {artifact.gates_path} are malformed.")
    result: dict[str, dict[str, object]] = {}
    for comparison_id, raw in comparisons.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Battery comparison {comparison_id!r} is malformed.")
        metric = raw.get("metric")
        legs = raw.get("legs")
        if not isinstance(metric, str) or metric not in ORIGIN_BATTERY_METRIC_KINDS:
            raise ValueError(
                f"Battery comparison {comparison_id!r} has unknown metric {metric!r}."
            )
        if metric == "monetary_sign_separated":
            expected_leg_names = set(
                _string_list(
                    _battery_receipt_config().get("monetary_leg_names"),
                    label="Phase-P monetary battery leg names",
                )
            )
            if not isinstance(legs, Mapping) or set(legs) != expected_leg_names:
                raise ValueError(
                    f"Battery monetary comparison {comparison_id!r} must contain "
                    f"exactly legs {sorted(expected_leg_names)}."
                )
        elif "legs" in raw:
            raise ValueError(
                f"Battery non-monetary comparison {comparison_id!r} cannot contain "
                "physical sub-legs."
            )
        if "legs" in raw:
            if not isinstance(legs, Mapping) or not legs:
                raise ValueError(
                    f"Battery comparison {comparison_id!r} legs must be a "
                    "nonempty object."
                )
            for leg_name, leg in legs.items():
                if not isinstance(leg, Mapping):
                    raise ValueError(
                        f"Battery leg {comparison_id}/{leg_name} is malformed."
                    )
                leg_id = f"{comparison_id}/{leg_name}"
                leg_failures = [
                    failure for failure in failures if failure.startswith(f"{leg_id}:")
                ]
                leg_status = leg.get("status")
                operational_status = (
                    raw.get("status") if leg_status is None else leg_status
                )
                result[leg_id] = {
                    "comparison_id": comparison_id,
                    "leg": leg_name,
                    "metric": metric,
                    "raw_status": operational_status,
                    "gate_status": _gate_leg_status(
                        operational_status,
                        leg_failures,
                    ),
                    "failure_messages": leg_failures,
                    "ratio": leg.get("incidence_ratio_acs_over_asec"),
                    "asec_incidence": leg.get("asec_incidence"),
                    "acs_incidence": leg.get("acs_incidence"),
                    "nonzero_rows": leg.get("nonzero_rows"),
                    "quantile_envelope_distance": leg.get("quantile_envelope_distance"),
                    "total_variation_distance": None,
                }
        else:
            leg_id = str(comparison_id)
            leg_failures = [
                failure for failure in failures if failure.startswith(f"{leg_id}:")
            ]
            operational_status = raw.get("status")
            result[leg_id] = {
                "comparison_id": comparison_id,
                "leg": None,
                "metric": metric,
                "raw_status": operational_status,
                "gate_status": _gate_leg_status(
                    operational_status,
                    leg_failures,
                ),
                "failure_messages": leg_failures,
                "ratio": raw.get("incidence_ratio_acs_over_asec"),
                "asec_incidence": raw.get("asec_incidence"),
                "acs_incidence": raw.get("acs_incidence"),
                "nonzero_rows": raw.get("nonzero_rows"),
                "quantile_envelope_distance": raw.get("quantile_envelope_distance"),
                "total_variation_distance": raw.get("total_variation_distance"),
            }
    mapped_failures = [
        failure for leg in result.values() for failure in leg["failure_messages"]
    ]
    if Counter(mapped_failures) != Counter(failures):
        missing = list((Counter(failures) - Counter(mapped_failures)).elements())
        repeated = list((Counter(mapped_failures) - Counter(failures)).elements())
        raise ValueError(
            f"Battery failures in {artifact.gates_path} did not map exactly once "
            f"to physical legs; unmatched={missing}, multiply_mapped={repeated}."
        )
    return result


def _gate_leg_status(
    operational_status: object,
    failure_messages: Sequence[str],
) -> str:
    allowed = {
        "tested",
        "dead_comparison",
        "absent_on_both_origins",
        "insufficient_support",
    }
    if not isinstance(operational_status, str) or operational_status not in allowed:
        raise ValueError(
            f"Unknown battery operational status {operational_status!r}; refusing "
            "to infer a gate status."
        )
    if failure_messages or operational_status == "dead_comparison":
        return "FAIL"
    if operational_status in {"absent_on_both_origins", "insufficient_support"}:
        return "NOT_TESTED"
    if operational_status == "tested":
        return "PASS"
    raise RuntimeError("Validated battery status did not map to a gate status.")


def _band_status(ratio: object, *, lower: float, upper: float) -> str:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float, str)):
        return "not_ratio_comparable"
    try:
        numeric = float(ratio)
    except ValueError:
        return "not_ratio_comparable"
    if math.isnan(numeric):
        return "not_ratio_comparable"
    return "in_band" if lower <= numeric <= upper else "out_of_band"


def battery_delta(
    baseline: Artifact,
    candidate: Artifact,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    baseline_legs = _battery_legs(baseline)
    candidate_legs = _battery_legs(candidate)
    if set(baseline_legs) != set(candidate_legs):
        raise ValueError(
            "Baseline/candidate battery leg sets differ: "
            f"missing_candidate={sorted(set(baseline_legs) - set(candidate_legs))}, "
            f"extra_candidate={sorted(set(candidate_legs) - set(baseline_legs))}."
        )
    lower, upper = map(float, PHASE_P_CONFIG["verdict"]["ratio_band"])
    counts = {
        "total_legs": 0,
        "ratio_comparable_legs": 0,
        "entered_band": 0,
        "left_band": 0,
        "unchanged": 0,
        "unchanged_in_band": 0,
        "unchanged_out_of_band": 0,
        "unchanged_not_ratio_comparable": 0,
    }
    rows: list[dict[str, object]] = []
    for leg_id in sorted(baseline_legs):
        base = baseline_legs[leg_id]
        cand = candidate_legs[leg_id]
        if base["metric"] != cand["metric"]:
            raise ValueError(f"Battery metric changed for {leg_id!r}.")
        base_status = _band_status(base["ratio"], lower=lower, upper=upper)
        candidate_status = _band_status(cand["ratio"], lower=lower, upper=upper)
        counts["total_legs"] += 1
        transition: str
        if base_status != "in_band" and candidate_status == "in_band":
            transition = "entered_band"
        elif base_status == "in_band" and candidate_status != "in_band":
            transition = "left_band"
        elif base_status == candidate_status == "in_band":
            transition = "unchanged_in_band"
        elif base_status == candidate_status == "out_of_band":
            transition = "unchanged_out_of_band"
        else:
            transition = "unchanged_not_ratio_comparable"
        counts[transition] += 1
        if transition.startswith("unchanged_"):
            counts["unchanged"] += 1
        if base_status != "not_ratio_comparable" and candidate_status != (
            "not_ratio_comparable"
        ):
            counts["ratio_comparable_legs"] += 1
        rows.append(
            {
                "leg_id": leg_id,
                "metric": base["metric"],
                "baseline": {
                    **base,
                    "band_status": base_status,
                },
                "candidate": {
                    **cand,
                    "band_status": candidate_status,
                },
                "gate_status_transition": (
                    f"{base['gate_status']}->{cand['gate_status']}"
                ),
                "transition": transition,
            }
        )
    return rows, counts


def participation_incidence(
    baseline: Artifact,
    candidate: Artifact,
) -> list[dict[str, object]]:
    base = _battery_legs(baseline)
    cand = _battery_legs(candidate)
    rows: list[dict[str, object]] = []
    for leg_id in sorted(base):
        if "/model_required_boolean/" not in leg_id:
            continue
        if leg_id not in cand:
            raise ValueError(f"Candidate battery lacks participation leg {leg_id!r}.")
        rows.append(
            {
                "leg_id": leg_id,
                "baseline": {
                    "asec_incidence": base[leg_id]["asec_incidence"],
                    "acs_incidence": base[leg_id]["acs_incidence"],
                    "incidence_ratio_acs_over_asec": base[leg_id]["ratio"],
                    "nonzero_rows": base[leg_id]["nonzero_rows"],
                },
                "candidate": {
                    "asec_incidence": cand[leg_id]["asec_incidence"],
                    "acs_incidence": cand[leg_id]["acs_incidence"],
                    "incidence_ratio_acs_over_asec": cand[leg_id]["ratio"],
                    "nonzero_rows": cand[leg_id]["nonzero_rows"],
                },
            }
        )
    if not rows:
        raise ValueError("Battery contains no model_required_boolean targets.")
    return rows


def _align_measurements(
    baseline_ids: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_index = pd.Index(baseline_ids)
    candidate_index = pd.Index(candidate_ids)
    if base_index.has_duplicates or candidate_index.has_duplicates:
        raise ValueError("Paired measurement stable donor ids must be unique.")
    missing_candidate = base_index.difference(candidate_index, sort=True)
    extra_candidate = candidate_index.difference(base_index, sort=True)
    if len(missing_candidate) or len(extra_candidate):
        raise ValueError(
            "Paired measurement stable donor id sets differ: "
            f"missing_candidate={missing_candidate.tolist()}, "
            f"extra_candidate={extra_candidate.tolist()}."
        )
    if base_index.empty:
        raise ValueError("Paired measurement stable donor id sets are empty.")
    ordered_ids = base_index.sort_values()
    base_positions = base_index.get_indexer(ordered_ids)
    candidate_positions = candidate_index.get_indexer(ordered_ids)
    if (base_positions < 0).any() or (candidate_positions < 0).any():
        raise RuntimeError("Paired donor alignment produced missing positions.")
    return base_positions, candidate_positions


def _cluster_sums(
    clusters: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    *,
    cluster_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = np.bincount(clusters, weights=weights, minlength=cluster_count)
    numerator = np.bincount(
        clusters,
        weights=weights * values,
        minlength=cluster_count,
    )
    return numerator, denominator


def _bootstrap_ci(values: np.ndarray) -> list[float]:
    bootstrap = PHASE_P_CONFIG["bootstrap"]
    assert isinstance(bootstrap, Mapping)
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    return [
        float(
            np.quantile(
                values,
                alpha,
                method=str(bootstrap["percentile_method"]),
            )
        ),
        float(
            np.quantile(
                values,
                1.0 - alpha,
                method=str(bootstrap["percentile_method"]),
            )
        ),
    ]


def _validated_oos_measurement_vectors(
    measurement: OOSTargetMeasurement,
    *,
    arm: str,
    cluster_count: int,
) -> dict[str, np.ndarray]:
    """Return one arm's finite, aligned vectors for the paired estimand."""

    ids = np.asarray(measurement.ids, dtype=object)
    if ids.ndim != 1 or not len(ids):
        raise ValueError(
            f"Paired OOS {arm} ids for {measurement.key!r} must be a nonempty "
            "one-dimensional vector."
        )
    if pd.isna(ids).any():
        raise ValueError(f"Paired OOS {arm} ids for {measurement.key!r} contain nulls.")
    if measurement.holdout_rows != len(ids):
        raise ValueError(
            f"Paired OOS {arm} holdout count for {measurement.key!r} differs "
            f"from its retained vectors: {measurement.holdout_rows} != {len(ids)}."
        )

    result: dict[str, np.ndarray] = {"ids": ids}
    for name in ("weights", "truth", "losses"):
        try:
            values = np.asarray(getattr(measurement, name), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Paired OOS {arm} {name} for {measurement.key!r} must be numeric."
            ) from exc
        if values.ndim != 1 or len(values) != len(ids):
            raise ValueError(
                f"Paired OOS {arm} {name} for {measurement.key!r} must be a "
                f"one-dimensional vector of length {len(ids)}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Paired OOS {arm} {name} for {measurement.key!r} contain "
                "non-finite values."
            )
        result[name] = values
    if (result["weights"] <= 0.0).any():
        raise ValueError(
            f"Paired OOS {arm} weights for {measurement.key!r} must be "
            "strictly positive."
        )
    if (result["losses"] < 0.0).any():
        raise ValueError(
            f"Paired OOS {arm} losses for {measurement.key!r} must be nonnegative."
        )

    raw_clusters = np.asarray(measurement.clusters)
    if (
        raw_clusters.ndim != 1
        or len(raw_clusters) != len(ids)
        or not np.issubdtype(raw_clusters.dtype, np.integer)
    ):
        raise ValueError(
            f"Paired OOS {arm} clusters for {measurement.key!r} must be a "
            f"one-dimensional integer vector of length {len(ids)}."
        )
    clusters = raw_clusters.astype(np.int64, copy=False)
    if ((clusters < 0) | (clusters >= cluster_count)).any():
        raise ValueError(
            f"Paired OOS {arm} clusters for {measurement.key!r} must be in "
            f"[0, {cluster_count})."
        )
    result["clusters"] = clusters
    return result


def _paired_drift_diagnostics(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    absolute_tolerance: float,
    include_weight_totals: bool,
) -> dict[str, object]:
    """Describe arm drift without turning it into an acceptance threshold."""

    delta = candidate - baseline
    absolute = np.abs(delta)
    changed = absolute > absolute_tolerance
    result: dict[str, object] = {
        "absolute_tolerance": absolute_tolerance,
        "changed_rows": int(changed.sum()),
        "unchanged_rows": int((~changed).sum()),
        "changed_fraction": float(changed.mean()),
        "candidate_greater_rows": int((changed & (delta > 0.0)).sum()),
        "candidate_less_rows": int((changed & (delta < 0.0)).sum()),
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "mean_absolute_difference": float(absolute.mean()),
    }
    if include_weight_totals:
        relative = absolute / np.abs(baseline)
        baseline_total = float(baseline.sum())
        candidate_total = float(candidate.sum())
        result.update(
            {
                "maximum_relative_difference_from_baseline": float(
                    relative.max(initial=0.0)
                ),
                "baseline_total": baseline_total,
                "candidate_total": candidate_total,
                "candidate_minus_baseline_total": candidate_total - baseline_total,
                "candidate_minus_baseline_total_fraction": (
                    candidate_total / baseline_total - 1.0
                ),
            }
        )
    return result


def _paired_loss_comparison(
    baseline: OOSTargetMeasurement,
    candidate: OOSTargetMeasurement,
) -> dict[str, object]:
    if baseline.key != candidate.key:
        raise ValueError(
            f"Paired OOS measurement keys differ: {baseline.key!r} != "
            f"{candidate.key!r}."
        )
    if baseline.entity != candidate.entity:
        raise ValueError(
            f"Paired OOS entities differ for {baseline.key!r}: "
            f"{baseline.entity!r} != {candidate.entity!r}."
        )

    bootstrap = PHASE_P_CONFIG["bootstrap"]
    estimand = PHASE_P_CONFIG["paired_oos_estimand"]
    loss_config = PHASE_P_CONFIG["quantile_loss"]
    assert (
        isinstance(bootstrap, Mapping)
        and isinstance(estimand, Mapping)
        and isinstance(loss_config, Mapping)
    )
    if (
        loss_config.get("per_target_holdout_aggregation")
        != "arm_specific_scoring_weighted_mean"
        or loss_config.get("paired_delta") != "candidate_minus_baseline"
    ):
        raise ValueError("Unsupported Phase-P target quantile-loss aggregation.")
    cluster_count = int(bootstrap["clusters"])
    absolute_tolerance = float(estimand["drift_diagnostics_absolute_tolerance"])
    base_vectors = _validated_oos_measurement_vectors(
        baseline,
        arm="baseline",
        cluster_count=cluster_count,
    )
    candidate_vectors = _validated_oos_measurement_vectors(
        candidate,
        arm="candidate",
        cluster_count=cluster_count,
    )
    base_pos, cand_pos = _align_measurements(
        base_vectors["ids"], candidate_vectors["ids"]
    )
    base_weights = base_vectors["weights"][base_pos]
    candidate_weights = candidate_vectors["weights"][cand_pos]
    base_truth = base_vectors["truth"][base_pos]
    candidate_truth = candidate_vectors["truth"][cand_pos]
    base_loss = base_vectors["losses"][base_pos]
    candidate_loss = candidate_vectors["losses"][cand_pos]

    base_weight_total = float(base_weights.sum())
    candidate_weight_total = float(candidate_weights.sum())
    base_mean = float(np.dot(base_weights, base_loss) / base_weight_total)
    candidate_mean = float(
        np.dot(candidate_weights, candidate_loss) / candidate_weight_total
    )
    delta = candidate_mean - base_mean

    base_clusters = base_vectors["clusters"][base_pos]
    candidate_clusters = candidate_vectors["clusters"][cand_pos]
    if not np.array_equal(base_clusters, candidate_clusters):
        raise ValueError(f"Paired OOS bootstrap clusters changed for {baseline.key!r}.")
    base_num, base_denominator = _cluster_sums(
        base_clusters,
        base_weights,
        base_loss,
        cluster_count=cluster_count,
    )
    cand_num, cand_denominator = _cluster_sums(
        base_clusters,
        candidate_weights,
        candidate_loss,
        cluster_count=cluster_count,
    )
    base_active = base_denominator > 0.0
    candidate_active = cand_denominator > 0.0
    if not np.array_equal(base_active, candidate_active):
        raise RuntimeError("Paired OOS active bootstrap clusters changed unexpectedly.")
    active = np.flatnonzero(base_active)
    if len(active) < 2:
        raise ValueError(f"OOS target {baseline.key!r} has fewer than two clusters.")
    rng = np.random.default_rng(
        _seed("phase-p-oos-bootstrap", bootstrap["seed"], baseline.key)
    )
    replicate_count = int(bootstrap["replicates"])
    replicate_delta = np.empty(replicate_count, dtype=np.float64)
    for start in range(0, replicate_count, 256):
        size = min(256, replicate_count - start)
        sampled = rng.integers(0, len(active), size=(size, len(active)))
        selected = active[sampled]
        base_denom = base_denominator[selected].sum(axis=1)
        candidate_denom = cand_denominator[selected].sum(axis=1)
        replicate_delta[start : start + size] = (
            cand_num[selected].sum(axis=1) / candidate_denom
            - base_num[selected].sum(axis=1) / base_denom
        )
    ci = _bootstrap_ci(replicate_delta)
    return {
        "estimand": str(estimand["name"]),
        "quantile_grid": [float(value) for value in PHASE_P_CONFIG["quantiles"]],
        "quantile_loss_aggregation": dict(loss_config),
        "paired_rows": len(base_pos),
        "bootstrap_nonempty_clusters": len(active),
        "bootstrap_pairing": str(estimand["bootstrap_pairing"]),
        "baseline_mean_pinball_loss": base_mean,
        "candidate_mean_pinball_loss": candidate_mean,
        "candidate_minus_baseline": delta,
        "candidate_minus_baseline_ci": ci,
        "statistically_significant_worsening": ci[0] > 0.0,
        "weight_drift": _paired_drift_diagnostics(
            base_weights,
            candidate_weights,
            absolute_tolerance=absolute_tolerance,
            include_weight_totals=True,
        ),
        "truth_drift": _paired_drift_diagnostics(
            base_truth,
            candidate_truth,
            absolute_tolerance=absolute_tolerance,
            include_weight_totals=False,
        ),
    }


def _target_metric_index(artifact: Artifact) -> dict[tuple[str, str, str], str]:
    result: dict[tuple[str, str, str], str] = {}
    for leg in _battery_legs(artifact).values():
        metric = leg.get("metric")
        if not isinstance(metric, str) or metric not in ORIGIN_BATTERY_METRIC_KINDS:
            raise ValueError(
                f"Battery comparison {leg['comparison_id']!r} has unknown "
                f"metric {metric!r}."
            )
        comparison_id = str(leg["comparison_id"])
        base = comparison_id.split("[", 1)[0]
        pieces = base.split("/", 2)
        if len(pieces) == 3:
            key = (pieces[0], pieces[1], pieces[2])
            previous = result.get(key)
            if previous is not None and previous != metric:
                raise ValueError(
                    f"Battery target {comparison_id!r} has conflicting metrics "
                    f"{previous!r} and {metric!r}."
                )
            result[key] = metric
    return result


def _money_amount_config() -> Mapping[str, object]:
    value = PHASE_P_CONFIG.get("money_amount_classification")
    if not isinstance(value, Mapping):  # pragma: no cover - fixed config
        raise TypeError("Phase-P money amount classification must be an object.")
    return value


def _validated_money_amount_keys(
    target_metrics: Mapping[str, str],
) -> frozenset[str]:
    """Validate and return the explicit money-only subset of numeric targets."""

    config = _money_amount_config()
    money = set(
        _string_list(
            config.get("money_amount_target_keys"),
            label="Phase-P money amount target keys",
        )
    )
    numeric_non_money = set(
        _string_list(
            config.get("generic_numeric_non_money_target_keys"),
            label="Phase-P generic numeric non-money target keys",
        )
    )
    overlap = money.intersection(numeric_non_money)
    if overlap:  # pragma: no cover - fixed config
        raise ValueError(
            f"Phase-P money and non-money target authorities overlap: {sorted(overlap)}."
        )
    unknown_metrics = {
        key: metric
        for key, metric in target_metrics.items()
        if metric not in ORIGIN_BATTERY_METRIC_KINDS
    }
    if unknown_metrics:
        raise ValueError(
            f"Phase-P OOS target metrics contain unknown kinds: {unknown_metrics}."
        )
    expected_changed_count = int(config["changed_target_count"])
    if len(target_metrics) != expected_changed_count:
        raise ValueError(
            "Phase-P changed-target classification count differs from its "
            f"authority: observed={len(target_metrics)}, "
            f"expected={expected_changed_count}."
        )
    generic_numeric = {
        key
        for key, metric in target_metrics.items()
        if metric == "monetary_sign_separated"
    }
    declared_numeric = money.union(numeric_non_money)
    if generic_numeric != declared_numeric:
        raise ValueError(
            "Phase-P generic numeric target partition differs from its authority: "
            f"missing={sorted(generic_numeric - declared_numeric)}, "
            f"extra={sorted(declared_numeric - generic_numeric)}."
        )
    expected_numeric_count = int(config["generic_numeric_metric_target_count"])
    expected_money_count = int(config["money_amount_target_count"])
    if (
        len(generic_numeric) != expected_numeric_count
        or len(money) != expected_money_count
    ):
        raise ValueError(
            "Phase-P money classification counts differ from their authority: "
            f"generic_numeric={len(generic_numeric)}/{expected_numeric_count}, "
            f"money={len(money)}/{expected_money_count}."
        )
    return frozenset(money)


def compare_oos(
    baseline: ArmMeasurement,
    candidate: ArmMeasurement,
    *,
    metric_index: Mapping[tuple[str, str, str], str],
) -> list[dict[str, object]]:
    if set(baseline.targets) != set(candidate.targets):
        raise ValueError(
            "Baseline/candidate OOS target sets differ: "
            f"missing_candidate={sorted(set(baseline.targets) - set(candidate.targets))}, "
            f"extra_candidate={sorted(set(candidate.targets) - set(baseline.targets))}."
        )
    metrics_by_key: dict[str, str] = {}
    for key, measurement in baseline.targets.items():
        battery_family = measurement.family.split("__batch_", 1)[0]
        metric = metric_index.get(
            (measurement.entity, battery_family, measurement.target)
        )
        if metric is None:
            raise ValueError(
                f"Battery does not classify OOS target {measurement.entity}/"
                f"{measurement.family}/{measurement.target}."
            )
        metrics_by_key[key] = metric
    money_amount_keys = _validated_money_amount_keys(metrics_by_key)
    rows: list[dict[str, object]] = []
    for key in sorted(baseline.targets):
        base = baseline.targets[key]
        cand = candidate.targets[key]
        metric = metrics_by_key[key]
        paired = _paired_loss_comparison(base, cand)
        rows.append(
            {
                "key": key,
                "family_id": base.family_id,
                "stage": base.stage,
                "entity": base.entity,
                "family": base.family,
                "target": base.target,
                "metric": metric,
                "is_money_amount": key in money_amount_keys,
                "baseline": {
                    "target_position": base.target_position,
                    "chain_predecessors": list(base.chain_predecessors),
                    "predictor_set": base.predictor_set,
                    "resolved_predictors": list(base.resolved_predictors),
                    "unavailable_optional_predictors": list(
                        base.unavailable_optional_predictors
                    ),
                    "predictor_source_bindings": base.predictor_source_bindings,
                    "training_rows": base.training_rows,
                    "holdout_rows": base.holdout_rows,
                },
                "candidate": {
                    "target_position": cand.target_position,
                    "chain_predecessors": list(cand.chain_predecessors),
                    "predictor_set": cand.predictor_set,
                    "resolved_predictors": list(cand.resolved_predictors),
                    "unavailable_optional_predictors": list(
                        cand.unavailable_optional_predictors
                    ),
                    "predictor_source_bindings": cand.predictor_source_bindings,
                    "training_rows": cand.training_rows,
                    "holdout_rows": cand.holdout_rows,
                },
                "paired": paired,
            }
        )
    return rows


def _paired_incidence_origin(
    baseline: IncidenceMeasurement,
    candidate: IncidenceMeasurement,
) -> dict[str, object]:
    base_pos, cand_pos = _align_measurements(baseline.ids, candidate.ids)
    weights = baseline.weights[base_pos]
    if not np.allclose(weights, candidate.weights[cand_pos], rtol=0.0, atol=1e-10):
        raise ValueError(f"CHAMPVA {baseline.origin} weights changed between arms.")
    base_values = baseline.values[base_pos]
    candidate_values = candidate.values[cand_pos]
    if (
        not np.isin(base_values, [0.0, 1.0]).all()
        or not np.isin(candidate_values, [0.0, 1.0]).all()
    ):
        raise ValueError("CHAMPVA incidence inputs must be physical booleans.")
    base_clusters = baseline.clusters[base_pos]
    if not np.array_equal(base_clusters, candidate.clusters[cand_pos]):
        raise ValueError(f"CHAMPVA {baseline.origin} clusters changed between arms.")
    bootstrap = PHASE_P_CONFIG["bootstrap"]
    assert isinstance(bootstrap, Mapping)
    cluster_count = int(bootstrap["clusters"])
    base_num, denominator = _cluster_sums(
        base_clusters,
        weights,
        base_values,
        cluster_count=cluster_count,
    )
    candidate_num, candidate_denominator = _cluster_sums(
        base_clusters,
        weights,
        candidate_values,
        cluster_count=cluster_count,
    )
    if not np.array_equal(denominator, candidate_denominator):
        raise RuntimeError("CHAMPVA paired cluster denominator changed.")
    active = np.flatnonzero(denominator > 0)
    return {
        "rows": len(base_pos),
        "active_clusters": active,
        "baseline_numerator": base_num,
        "candidate_numerator": candidate_num,
        "denominator": denominator,
        "baseline_incidence": float(np.dot(weights, base_values) / weights.sum()),
        "candidate_incidence": float(np.dot(weights, candidate_values) / weights.sum()),
    }


def champva_ratio_bootstrap(
    baseline: ArmMeasurement,
    candidate: ArmMeasurement,
    *,
    baseline_ratio_receipt: float,
    candidate_ratio_receipt: float,
) -> dict[str, object]:
    origins = {
        origin: _paired_incidence_origin(
            baseline.champva_by_origin[origin],
            candidate.champva_by_origin[origin],
        )
        for origin in (_ASEC_CHANNEL, _ACS_CHANNEL)
    }
    base_ratio = (
        origins[_ACS_CHANNEL]["baseline_incidence"]
        / origins[_ASEC_CHANNEL]["baseline_incidence"]
    )
    candidate_ratio = (
        origins[_ACS_CHANNEL]["candidate_incidence"]
        / origins[_ASEC_CHANNEL]["candidate_incidence"]
    )
    if not math.isclose(base_ratio, baseline_ratio_receipt, rel_tol=1e-10, abs_tol=0.0):
        raise ValueError(
            "Baseline CHAMPVA checkpoint ratio does not reproduce its gates receipt: "
            f"checkpoint={base_ratio}, receipt={baseline_ratio_receipt}."
        )
    if not math.isclose(
        candidate_ratio,
        candidate_ratio_receipt,
        rel_tol=1e-10,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Candidate CHAMPVA checkpoint ratio does not reproduce its gates receipt: "
            f"checkpoint={candidate_ratio}, receipt={candidate_ratio_receipt}."
        )

    bootstrap = PHASE_P_CONFIG["bootstrap"]
    assert isinstance(bootstrap, Mapping)
    replicate_count = int(bootstrap["replicates"])
    rng = np.random.default_rng(_seed("phase-p-champva-bootstrap", bootstrap["seed"]))
    replicate: dict[str, dict[str, np.ndarray]] = {}
    for origin, measured in origins.items():
        active = measured["active_clusters"]
        if len(active) < 2:
            raise ValueError(f"CHAMPVA {origin} has fewer than two clusters.")
        base_incidence = np.empty(replicate_count, dtype=np.float64)
        candidate_incidence = np.empty(replicate_count, dtype=np.float64)
        for start in range(0, replicate_count, 256):
            size = min(256, replicate_count - start)
            sampled = rng.integers(0, len(active), size=(size, len(active)))
            selected = active[sampled]
            denominator = measured["denominator"][selected].sum(axis=1)
            base_incidence[start : start + size] = (
                measured["baseline_numerator"][selected].sum(axis=1) / denominator
            )
            candidate_incidence[start : start + size] = (
                measured["candidate_numerator"][selected].sum(axis=1) / denominator
            )
        replicate[origin] = {
            "baseline": base_incidence,
            "candidate": candidate_incidence,
        }
    finite = (replicate[_ASEC_CHANNEL]["baseline"] > 0) & (
        replicate[_ASEC_CHANNEL]["candidate"] > 0
    )
    if int(finite.sum()) < math.ceil(0.95 * replicate_count):
        raise ValueError(
            "Too many CHAMPVA bootstrap replicates have zero ASEC incidence."
        )
    base_rep_ratio = (
        replicate[_ACS_CHANNEL]["baseline"][finite]
        / replicate[_ASEC_CHANNEL]["baseline"][finite]
    )
    candidate_rep_ratio = (
        replicate[_ACS_CHANNEL]["candidate"][finite]
        / replicate[_ASEC_CHANNEL]["candidate"][finite]
    )
    improvement = base_rep_ratio - candidate_rep_ratio
    return {
        "baseline_ratio": base_ratio,
        "candidate_ratio": candidate_ratio,
        "baseline_minus_candidate": base_ratio - candidate_ratio,
        "baseline_minus_candidate_ci": _bootstrap_ci(improvement),
        "finite_bootstrap_replicates": int(finite.sum()),
        "origin_rows": {
            origin: int(measured["rows"]) for origin, measured in origins.items()
        },
    }


def _artifact_receipt(artifact: Artifact) -> dict[str, object]:
    external = _external_metadata(artifact)
    identity = external.get("identity")
    operational: dict[str, object] | None = None
    if artifact.operational_receipts_path.is_file():
        operational_payload = json.loads(artifact.operational_receipts_path.read_text())
        checkpoint = (
            operational_payload.get("checkpoint")
            if isinstance(operational_payload, Mapping)
            else None
        )
        operational = {
            "path": str(artifact.operational_receipts_path),
            "sha256": _sha256_file(artifact.operational_receipts_path),
            "checkpoint_binding": checkpoint,
        }
    return {
        "artifact_dir": str(artifact.root),
        "gates_json": str(artifact.gates_path),
        "gates_json_sha256": _sha256_file(artifact.gates_path),
        "simulated_checkpoint_h5": str(artifact.checkpoint_path),
        "simulated_checkpoint_size_bytes": artifact.checkpoint_path.stat().st_size,
        "stacked_identity": artifact.checkpoint_path.parent.name,
        "checkpoint_identity_sha256": external.get("identity_sha256"),
        "operational_receipts": operational,
        "run_identity": identity if isinstance(identity, Mapping) else None,
    }


def _gate_implementation_receipt() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": _sha256_file(path)}


def measurement_failure_verdict(
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
    error: Exception,
) -> dict[str, object]:
    """Return a machine-decidable FAIL when the required sweep cannot finish."""

    context = error.context if isinstance(error, GateMeasurementError) else None
    return {
        "schema_version": 1,
        "artifact_kind": "phase_p_predictor_set_oos_acceptance_verdict",
        "verdict": "FAIL",
        "passed": False,
        "measurement_complete": False,
        "failing_clauses": ["measurement_integrity"],
        "rule": {
            "pass_iff": PHASE_P_CONFIG["verdict"]["pass_iff"],
            "measurement_precondition": (
                "All declared family/target OOS measurements, battery legs, and "
                "paired bootstrap inputs must be available before clauses a-c "
                "can be evaluated; otherwise the verdict is FAIL."
            ),
        },
        "clauses": {
            "measurement_integrity": {
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "context": context,
            },
            "a_champva": {"passed": None, "evaluated": False},
            "b_money_oos": {"passed": None, "evaluated": False},
            "c_no_battery_leg_leaves_band": {
                "passed": None,
                "evaluated": False,
            },
        },
        "config": PHASE_P_CONFIG,
        "gate_implementation": _gate_implementation_receipt(),
        "artifacts": {
            "baseline": {"artifact_dir": str(Path(baseline_path).resolve())},
            "candidate": {"artifact_dir": str(Path(candidate_path).resolve())},
        },
    }


def build_verdict(
    *,
    baseline: Artifact,
    candidate: Artifact,
    oos_rows: list[dict[str, object]],
    battery_rows: list[dict[str, object]],
    battery_counts: dict[str, int],
    participation_rows: list[dict[str, object]],
    champva_ci: dict[str, object],
    family_specs: Sequence[FamilySpec],
) -> dict[str, object]:
    rule = PHASE_P_CONFIG["verdict"]
    assert isinstance(rule, Mapping)
    lower, upper = map(float, rule["ratio_band"])
    baseline_ratio = float(champva_ci["baseline_ratio"])
    candidate_ratio = float(champva_ci["candidate_ratio"])
    improvement_ci = list(champva_ci["baseline_minus_candidate_ci"])
    enters_band = lower <= candidate_ratio <= upper
    denominator = baseline_ratio - upper
    progress = (
        (baseline_ratio - candidate_ratio) / denominator if denominator > 0 else None
    )
    progress_pass = (
        progress is not None
        and math.isfinite(progress)
        and progress >= float(rule["champva_minimum_progress_to_band_ceiling"])
        and float(improvement_ci[0]) > 0.0
    )
    clause_a = enters_band or progress_pass

    money_rows = [row for row in oos_rows if row["is_money_amount"]]
    money_worsened = [
        str(row["key"])
        for row in money_rows
        if row["paired"]["statistically_significant_worsening"]
    ]
    clause_b = bool(money_rows) and not money_worsened
    clause_c = battery_counts["left_band"] == 0
    clauses = {
        "a_champva": {
            "passed": clause_a,
            "candidate_enters_band": enters_band,
            "progress_from_baseline_to_band_ceiling": progress,
            "minimum_progress": rule["champva_minimum_progress_to_band_ceiling"],
            "improvement_ci_excludes_zero": float(improvement_ci[0]) > 0.0,
            "baseline_minus_candidate_ratio_ci": improvement_ci,
        },
        "b_money_oos": {
            "passed": clause_b,
            "money_target_count": len(money_rows),
            "significantly_worsened_targets": money_worsened,
            "worsening_definition": rule["money_worsening"],
        },
        "c_no_battery_leg_leaves_band": {
            "passed": clause_c,
            "leaving_band_count": battery_counts["left_band"],
        },
    }
    passed = all(bool(clause["passed"]) for clause in clauses.values())
    failing = [name for name, clause in clauses.items() if not clause["passed"]]
    return {
        "schema_version": 1,
        "artifact_kind": "phase_p_predictor_set_oos_acceptance_verdict",
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "measurement_complete": True,
        "failing_clauses": failing,
        "rule": {
            "pass_iff": rule["pass_iff"],
            "a": (
                "CHAMPVA ratio enters [0.8, 1.25], or reaches at least 50% "
                "of the baseline-to-1.25 distance with the paired bootstrap "
                "improvement CI strictly above zero."
            ),
            "b": (
                "No monetary target has a candidate-minus-baseline OOS "
                "pinball-loss bootstrap CI strictly above zero."
            ),
            "c": "No ratio-bearing battery leg leaves [0.8, 1.25].",
        },
        "clauses": clauses,
        "config": PHASE_P_CONFIG,
        "gate_implementation": _gate_implementation_receipt(),
        "artifacts": {
            "baseline": _artifact_receipt(baseline),
            "candidate": _artifact_receipt(candidate),
        },
        "family_count": len(family_specs),
        "target_count": len(oos_rows),
        "quantile_loss_definition": {
            "quantile_grid": [float(value) for value in PHASE_P_CONFIG["quantiles"]],
            "aggregation": PHASE_P_CONFIG["quantile_loss"],
        },
        "families": [
            {
                **{
                    key: value
                    for key, value in asdict(spec).items()
                    if key not in {"baseline_predictor_set", "candidate_predictor_set"}
                },
                "baseline_predictor_set": {
                    "required": list(spec.baseline_predictor_set.required),
                    "optional": list(spec.baseline_predictor_set.optional),
                },
                "candidate_predictor_set": {
                    "required": list(spec.candidate_predictor_set.required),
                    "optional": list(spec.candidate_predictor_set.optional),
                },
            }
            for spec in family_specs
        ],
        "oos_pinball_loss": oos_rows,
        "by_origin_battery": {
            "counts": battery_counts,
            "legs": battery_rows,
        },
        "participation_incidence": participation_rows,
        "champva_ratio_bootstrap": champva_ci,
    }


def run_gate(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    family_specs: Sequence[FamilySpec] | None = None,
) -> dict[str, object]:
    baseline = resolve_artifact(baseline_path)
    candidate = resolve_artifact(candidate_path)
    specs = (
        tuple(family_specs)
        if family_specs is not None
        else production_family_specs(baseline, candidate)
    )
    battery_rows, battery_counts = battery_delta(baseline, candidate)
    participation_rows = participation_incidence(baseline, candidate)
    baseline_evidence = _load_early_execution_evidence(
        baseline,
        arm="baseline",
        family_specs=specs,
    )
    candidate_evidence = _load_early_execution_evidence(
        candidate,
        arm="candidate",
        family_specs=specs,
    )
    baseline_measurement = measure_arm(
        baseline,
        arm="baseline",
        family_specs=specs,
        execution_evidence=baseline_evidence,
    )
    candidate_measurement = measure_arm(
        candidate,
        arm="candidate",
        family_specs=specs,
        execution_evidence=candidate_evidence,
    )
    metric_index = _target_metric_index(baseline)
    oos_rows = compare_oos(
        baseline_measurement,
        candidate_measurement,
        metric_index=metric_index,
    )
    champva_leg = str(PHASE_P_CONFIG["verdict"]["champva_leg"])
    base_champva = next(row for row in battery_rows if row["leg_id"] == champva_leg)
    champva_ci = champva_ratio_bootstrap(
        baseline_measurement,
        candidate_measurement,
        baseline_ratio_receipt=float(base_champva["baseline"]["ratio"]),
        candidate_ratio_receipt=float(base_champva["candidate"]["ratio"]),
    )
    return build_verdict(
        baseline=baseline,
        candidate=candidate,
        oos_rows=oos_rows,
        battery_rows=battery_rows,
        battery_counts=battery_counts,
        participation_rows=participation_rows,
        champva_ci=champva_ci,
        family_specs=specs,
    )


def _number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def render_markdown(payload: Mapping[str, object]) -> str:
    """Render the committed short report from the machine verdict."""

    if not payload.get("measurement_complete", True):
        failure = payload["clauses"]["measurement_integrity"]
        baseline = payload["artifacts"]["baseline"]
        candidate = payload["artifacts"]["candidate"]
        implementation = payload["gate_implementation"]
        return "\n".join(
            [
                "# Phase P predictor-sets acceptance gate",
                "",
                "## Verdict",
                "",
                "**FAIL**",
                "",
                "Failing clause: `measurement_integrity`.",
                "",
                f"Error: `{failure['error_type']}: {failure['error']}`.",
                "",
                f"Context: `{json.dumps(failure['context'], sort_keys=True)}`.",
                "",
                f"Baseline artifact: `{baseline['artifact_dir']}`.",
                "",
                f"Candidate artifact: `{candidate['artifact_dir']}`.",
                "",
                f"Gate implementation: `{implementation['path']}` "
                f"(SHA-256 `{implementation['sha256']}`).",
                "",
                "Clauses a-c were not evaluated because the declared OOS sweep "
                "did not complete. This gate fails closed; no tuning was attempted.",
                "",
            ]
        )

    artifacts = payload["artifacts"]
    baseline = artifacts["baseline"]
    candidate = artifacts["candidate"]
    clauses = payload["clauses"]
    clause_a = clauses["a_champva"]
    champva = payload["champva_ratio_bootstrap"]
    lines = [
        "# Phase P predictor-sets acceptance gate",
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']}**",
        "",
        f"Failing clauses: `{', '.join(payload['failing_clauses']) or 'none'}`.",
        "",
        "| Clause | Status | Receipt |",
        "|---|---:|---|",
    ]
    for name, clause in clauses.items():
        lines.append(
            f"| `{name}` | {'PASS' if clause['passed'] else 'FAIL'} | "
            f"`{baseline['gates_json']}`; `{candidate['gates_json']}`; "
            f"`{baseline['simulated_checkpoint_h5']}`; "
            f"`{candidate['simulated_checkpoint_h5']}` |"
        )
    lines.extend(
        [
            "",
            "Clause A CHAMPVA receipt: baseline ratio "
            f"{_number(champva['baseline_ratio'])}; candidate ratio "
            f"{_number(champva['candidate_ratio'])}; progress from the baseline "
            "toward the band ceiling "
            f"{_number(clause_a['progress_from_baseline_to_band_ceiling'])}; "
            "baseline-minus-candidate ratio 95% CI "
            f"[{_number(clause_a['baseline_minus_candidate_ratio_ci'][0])}, "
            f"{_number(clause_a['baseline_minus_candidate_ratio_ci'][1])}]; "
            f"candidate enters band: {str(clause_a['candidate_enters_band']).lower()}; "
            "improvement CI excludes zero: "
            f"{str(clause_a['improvement_ci_excludes_zero']).lower()}. Sources: "
            f"`{baseline['gates_json']}`; `{candidate['gates_json']}`; "
            f"`{baseline['simulated_checkpoint_h5']}`; "
            f"`{candidate['simulated_checkpoint_h5']}`.",
            "",
            "The rule is machine-decidable in the adjacent verdict JSON: PASS "
            "iff clauses a, b, and c all pass.",
            "",
            "The OOS delta is the full-artifact-arm weighted risk difference: "
            "stable donor ids and bootstrap cluster draws are paired, while "
            "each arm uses its own checkpoint outcomes and effective weights "
            "with a separate denominator. Weight and truth drift are reported "
            "per target in the adjacent verdict JSON.",
            "",
            "## Artifact receipts",
            "",
            "| Arm | Gates JSON | Simulated checkpoint | Operational receipts | "
            "Stacked identity |",
            "|---|---|---|---|---|",
            f"| baseline | `{baseline['gates_json']}` | "
            f"`{baseline['simulated_checkpoint_h5']}` | "
            f"`{(baseline['operational_receipts'] or {}).get('path', '—')}` | "
            f"`{baseline['stacked_identity']}` |",
            f"| candidate | `{candidate['gates_json']}` | "
            f"`{candidate['simulated_checkpoint_h5']}` | "
            f"`{(candidate['operational_receipts'] or {}).get('path', '—')}` | "
            f"`{candidate['stacked_identity']}` |",
            "",
            "## Per-target OOS pinball loss",
            "",
            "| Family | Target | Money | Baseline | Candidate | Delta | 95% CI | "
            "Train B/C | Holdout B/C | Weight drift | Truth drift | Sources |",
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["oos_pinball_loss"]:
        paired = row["paired"]
        ci = paired["candidate_minus_baseline_ci"]
        weight_drift = paired["weight_drift"]
        truth_drift = paired["truth_drift"]
        lines.append(
            f"| `{row['family_id']}` | `{row['target']}` | "
            f"{'yes' if row['is_money_amount'] else 'no'} | "
            f"{_number(paired['baseline_mean_pinball_loss'])} | "
            f"{_number(paired['candidate_mean_pinball_loss'])} | "
            f"{_number(paired['candidate_minus_baseline'])} | "
            f"[{_number(ci[0])}, {_number(ci[1])}] | "
            f"{row['baseline']['training_rows']}/{row['candidate']['training_rows']} | "
            f"{row['baseline']['holdout_rows']}/{row['candidate']['holdout_rows']} | "
            f"{weight_drift['changed_rows']}/{paired['paired_rows']} "
            f"(max abs {_number(weight_drift['maximum_absolute_difference'])}) | "
            f"{truth_drift['changed_rows']}/{paired['paired_rows']} "
            f"(max abs {_number(truth_drift['maximum_absolute_difference'])}) | "
            f"`{baseline['simulated_checkpoint_h5']}`; "
            f"`{candidate['simulated_checkpoint_h5']}` |"
        )
    battery = payload["by_origin_battery"]
    counts = battery["counts"]
    lines.extend(
        [
            "",
            "## By-origin battery leg delta",
            "",
            f"Entered band: {counts['entered_band']}; left band: "
            f"{counts['left_band']}; unchanged: {counts['unchanged']} "
            f"(in band: "
            f"{counts['unchanged_in_band']}; unchanged out of band: "
            f"{counts['unchanged_out_of_band']}; not ratio-comparable: "
            f"{counts['unchanged_not_ratio_comparable']}). Sources: "
            f"`{baseline['gates_json']}`; `{candidate['gates_json']}`.",
            "",
            "| Leg | Metric | Baseline gate/band | Baseline ratio | "
            "Baseline distance | Candidate gate/band | Candidate ratio | "
            "Candidate distance | Band transition | Sources |",
            "|---|---|---|---:|---:|---|---:|---:|---|---|",
        ]
    )
    for row in battery["legs"]:
        baseline_distance = (
            row["baseline"]["quantile_envelope_distance"]
            if row["baseline"]["quantile_envelope_distance"] is not None
            else row["baseline"]["total_variation_distance"]
        )
        candidate_distance = (
            row["candidate"]["quantile_envelope_distance"]
            if row["candidate"]["quantile_envelope_distance"] is not None
            else row["candidate"]["total_variation_distance"]
        )
        lines.append(
            f"| `{row['leg_id']}` | {row['metric']} | "
            f"{row['baseline']['gate_status']} / "
            f"{row['baseline']['band_status']} | "
            f"{_number(row['baseline']['ratio'])} | "
            f"{_number(baseline_distance)} | "
            f"{row['candidate']['gate_status']} / "
            f"{row['candidate']['band_status']} | "
            f"{_number(row['candidate']['ratio'])} | "
            f"{_number(candidate_distance)} | {row['transition']} | "
            f"`{baseline['gates_json']}`; `{candidate['gates_json']}` |"
        )
    lines.extend(
        [
            "",
            "## Participation incidence",
            "",
            "| Target leg | Baseline ASEC | Baseline ACS | Baseline ratio | "
            "Candidate ASEC | Candidate ACS | Candidate ratio | Sources |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["participation_incidence"]:
        base = row["baseline"]
        cand = row["candidate"]
        lines.append(
            f"| `{row['leg_id']}` | {_number(base['asec_incidence'])} | "
            f"{_number(base['acs_incidence'])} | "
            f"{_number(base['incidence_ratio_acs_over_asec'])} | "
            f"{_number(cand['asec_incidence'])} | "
            f"{_number(cand['acs_incidence'])} | "
            f"{_number(cand['incidence_ratio_acs_over_asec'])} | "
            f"`{baseline['gates_json']}`; `{candidate['gates_json']}` |"
        )
    lines.extend(
        [
            "",
            "## Configuration",
            "",
            f"Source: `{payload['gate_implementation']['path']}` "
            f"(SHA-256 `{payload['gate_implementation']['sha256']}`).",
            "",
            "```json",
            json.dumps(payload["config"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = run_gate(args.baseline, args.candidate)
    except Exception as error:
        payload = measurement_failure_verdict(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            error=error,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload))
    print(
        _canonical_json(
            {
                "verdict": payload["verdict"],
                "failing_clauses": payload["failing_clauses"],
                "output": str(args.output.resolve()),
                "markdown_output": (
                    None
                    if args.markdown_output is None
                    else str(args.markdown_output.resolve())
                ),
            }
        ),
        flush=True,
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
