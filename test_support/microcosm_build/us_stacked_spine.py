"""Stacked-spine pilot: stack, doctrine, gap-fill, gates (microcosm#578 revision).

The ratified increment-2 revision replaces the two-spine agreement construct
with ONE origin-labeled spine: ASEC plus a seeded ACS household sample,
cross-origin gap-fill with native predictors, a single PUF pass after
gap-fill, a pre-simulation completeness gate, and a by-origin battery with
per-family declared metrics.
"""

# ruff: noqa: F401

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import pickle
import sys
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import microcosm.build.us_runtime.acs_income_universe as universe_module
import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
import microcosm.build.us_runtime.multispine_pool as multispine_pool_module
import microcosm.build.us_runtime.post_transfer_calibration as post_transfer_calibration_runtime
import microcosm.build.us_runtime.puf_capital_gains_tail as tail_module
import microcosm.build.us_runtime.puf_support as puf_support_module
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
import microcosm.build.us_runtime.worker_identity as worker_identity_module
from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.gates import GateReport, GateResult
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime.acs_income_universe import (
    apply_acs_pums_earnings_universe_zeros,
)
from microcosm.build.us_runtime.acs_transfer import (
    AcsImputedInput,
    AcsTransferPattern,
    AcsTransferResult,
)
from microcosm.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerInputColumn,
)
from microcosm.build.us_runtime.multispine_pool import (
    PoolStageOutput,
    derive_multispine_pool_inputs,
    pool_transfer_target_families,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_TAX_UNIT_COLUMNS,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    PUF_CLONE_ATTACHMENT_MANIFEST_KEY,
    PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN,
    clone_us_frame_for_puf_support,
    finalize_us_puf_tax_detail_predictions,
    prepare_us_puf_tax_detail_chain_inputs,
    validate_puf_clone_attachment,
)
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.stacked_spine import (
    DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES,
    STACKED_PILOT_ACS_SAMPLE_FRACTION,
    STACKED_PILOT_ACS_SAMPLE_SEED,
    STACKED_SPINE_MANIFEST_KEY,
    AbsenceProof,
    GapFillAbsenceRule,
    GapFillDirection,
    OriginBatterySpec,
    assemble_stacked_spine,
    by_origin_battery,
    by_origin_battery_artifact_evidence,
    gap_fill_stacked_spine,
    run_stacked_puf_pass,
    sample_acs_households,
    stacked_completeness_gate,
    stacked_gap_fill_plan,
    stacked_gap_fill_producer_schedule_receipt,
    transfer_stacked_post_puf_inputs,
    validate_stacked_spine_frame,
)
from microcosm.build.us_runtime.support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    us_late_overlap_ownership_receipt,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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
        extra_person_columns={"acs_native_aggregate": 15.0, "WAGP": 15.0},
        extra_household_columns={
            "puma": "0600101",
            "TYPEHUGQ": 1,
            "tenure_type": "RENTED",
        },
        stratum="acs_2024_1yr",
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


def _s_corp_universe_fixture() -> Frame:
    cloned = _cloned_stacked_fixture()
    person = cloned.table("person").copy(deep=True)
    clone_column = support_clone_index_column("person")
    clone_index = person[clone_column]
    person["s_corp_income"] = np.where(clone_index.eq(0), np.nan, 0.0)
    first_clone = person.index[clone_index.eq(1)][0]
    person.loc[first_clone, clone_column] = 2
    tables = {entity: cloned.table(entity) for entity in cloned.entities}
    tables["person"] = person
    return Frame(
        tables,
        cloned.schema,
        {entity: cloned.weights_for(entity) for entity in cloned.weighted_entities},
        cloned.strata,
        mass_log=cloned.mass_log,
        metadata=cloned.metadata,
    )


def _cloned_acs_earnings_universe_fixture() -> Frame:
    """Apply receipted zeros to multi-child mixed and all-child ACS units."""

    cloned = _cloned_stacked_fixture()
    person = cloned.table("person")
    channel = person[support_channel_column("person")].astype(str)
    clone_index = person[support_clone_index_column("person")]
    acs_detail = channel.eq("acs") & clone_index.eq(1)
    detail_groups = list(
        person.loc[acs_detail].groupby("person_tax_unit_id", sort=False).groups.values()
    )
    mixed_detail = list(next(group for group in detail_groups if len(group) >= 3))
    all_child_detail = list(
        next(
            group
            for group in detail_groups
            if len(group) > 1 and list(group) != mixed_detail
        )
    )
    source_id = support_source_id_column("person")
    all_child_lineages = set(person.loc[all_child_detail, source_id])
    mixed_child_lineages = set(person.loc[mixed_detail[:-1], source_id])

    person["self_employment_income_before_lsr"] = 0.0
    person["WAGP"] = person["employment_income_before_lsr"]
    person["SEMP"] = person["self_employment_income_before_lsr"]
    acs = channel.eq("acs")
    person.loc[acs, "age"] = 40.0
    person.loc[acs, "employment_income_before_lsr"] = 100.0
    person.loc[acs, "self_employment_income_before_lsr"] = 10.0
    person.loc[acs, "WAGP"] = 100.0
    person.loc[acs, "SEMP"] = 10.0
    structural = acs & (
        person[source_id].isin(all_child_lineages)
        | person[source_id].isin(mixed_child_lineages)
    )
    person.loc[structural, "age"] = 12.0
    for column in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "WAGP",
        "SEMP",
    ):
        person.loc[structural, column] = np.nan
    return stacked_spine_module._materialize_stacked_acs_earnings_universe(cloned).frame


def _late_primary_entry(frame: Frame) -> Frame:
    """Materialize the declared DAG predecessor for direct primary tests."""

    return stacked_spine_module._materialize_stacked_acs_earnings_universe(frame).frame


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
    person["self_employment_income_before_lsr"] = 0.0
    person["pre_subsidy_rent"] = np.asarray(
        [12_000.0, 0.0, 0.0, 9_600.0, 0.0, 14_400.0]
    )
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
    person["WAGP"] = person["employment_income_before_lsr"]
    person["self_employment_income_before_lsr"] = 0.0
    person["SEMP"] = 0.0
    person["acs_interest_dividend_rental_income"] = np.asarray(
        [0.0, 400.0, 0.0, 150.0, 900.0, 0.0, 250.0, 0.0, 3_000.0, 120.0, 60.0]
    )
    household = frame.table("household").copy()
    household["TYPEHUGQ"] = np.asarray([1] * 9 + [2], dtype=np.int64)
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
            np.nan,
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


_CANONICAL_RENT_ABSENCE_RULE = GapFillAbsenceRule(
    **{
        field: getattr(stacked_gap_fill_plan()[1].recipient_absence_rules[0], field)
        for field in ("rule_id", "entity", "column", "selection", "reason")
    }
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
        name="asec_housing_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={"person": {"housing": ("pre_subsidy_rent",)}},
        recipient_absence_rules=(_CANONICAL_RENT_ABSENCE_RULE,),
    ),
)


def _registry_boolean_targets(
    surface: Mapping[str, Mapping[str, tuple[str, ...]]],
) -> set[tuple[str, str]]:
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    return {
        (entity, target)
        for entity, families in surface.items()
        for family, targets in families.items()
        for target in targets
        if registry[(entity, family, target, 0)] == "boolean_incidence"
    }


def _transferred_registry_boolean_targets() -> set[tuple[str, str]]:
    return _registry_boolean_targets(
        stacked_spine_module.CANONICAL_STACKED_GAP_FILL_SURFACE
    ) | _registry_boolean_targets(
        stacked_spine_module.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
    )


def _canonical_registry_checkpoint_frame() -> Frame:
    frame = _source_frame(
        household_ids=[1, 2, 3],
        weights=[1.0, 2.0, 3.0],
        stratum="registry_checkpoint",
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    registry = stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
    nullable_booleans = _transferred_registry_boolean_targets()
    string_categories = {"immigration_status_str", "ssn_card_type"}
    for (entity, _family, column, _clone_index), metric in sorted(registry.items()):
        table = tables[entity]
        if metric == "monetary_sign_separated":
            values: pd.Series | np.ndarray = np.asarray(
                [-1.25, 0.0, 2.5],
                dtype=np.float64,
            )
        elif metric == "boolean_incidence":
            if (entity, column) in nullable_booleans:
                values = pd.Series(
                    [True, pd.NA, False],
                    index=table.index,
                    dtype="boolean",
                )
            else:
                values = np.asarray([True, False, True], dtype=np.bool_)
        elif column in string_categories:
            values = pd.Series(
                ["A", pd.NA, "B"],
                index=table.index,
                dtype=CANONICAL_STRING_DTYPE,
            )
        else:
            assert metric == "categorical_tvd"
            values = np.asarray([1.0, np.nan, 3.0], dtype=np.float64)
        table[column] = values

    person = tables["person"]
    for column in ("is_female", "is_household_head"):
        person[column] = pd.Series(
            [True, False, True],
            index=person.index,
            dtype="boolean",
        )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
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


def _canonical_gap_fill_calibration_receipt() -> dict[str, object]:
    policy = (
        post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()
    )
    early_specs = {
        spec.key: spec
        for spec in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.stage == "early_gap_fill"
    }
    values = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    )
    weights = np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    reference = np.asarray([True] * 5 + [False] * 5)
    recipient = ~reference
    directions: dict[str, object] = {}
    for direction in stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN:
        target_keys = {
            f"{entity}/{family}/{target}"
            for entity, families in direction.target_families.items()
            for family, targets in families.items()
            for target in targets
        }
        calibrated_keys = sorted(target_keys & set(early_specs))
        target_receipts: dict[str, dict[str, object]] = {
            key: {
                "authorized_null_rows": 0,
                "imputed_rows": 0,
                "unmodeled_rows": 0,
                "residual_null_rows": 0,
            }
            for key in target_keys
        }
        for key in calibrated_keys:
            spec = early_specs[key]
            calibration_result = (
                post_transfer_calibration_runtime.calibrate_post_transfer_values(
                    values,
                    weights,
                    np.arange(1, len(values) + 1),
                    spec=spec,
                    reference_rows=reference,
                    recipient_rows=recipient,
                    mutable_rows=recipient,
                )
            )
            calibration = calibration_result.receipt
            scope = calibration["scope"]
            target_receipts[key]["post_transfer_calibration"] = {
                "stage": "early_gap_fill",
                "reference_selection": "asec_origin_clone_0",
                "recipient_selection": "acs_origin_clone_0",
                "mutable_selection": "recipient_null_before_nonnull_after",
                "reference_rows": scope["reference_rows"],
                "recipient_rows": scope["recipient_rows"],
                "mutable_rows": scope["mutable_rows"],
                "constraint": {"constraint": "none"},
                "context_binding": {
                    "scope": dict(scope),
                    "weights_sha256": calibration["weights"]["sha256"],
                    "live_output": {
                        "reference_rows": int(reference.sum()),
                        "recipient_rows": int(recipient.sum()),
                        "reference_entity_ids_sha256": (
                            stacked_spine_module._post_transfer_entity_ids_sha256(
                                np.arange(1, len(values) + 1)[reference]
                            )
                        ),
                        "recipient_entity_ids_sha256": (
                            stacked_spine_module._post_transfer_entity_ids_sha256(
                                np.arange(1, len(values) + 1)[recipient]
                            )
                        ),
                        "reference_output_values_sha256": (
                            stacked_spine_module._post_transfer_float64_sha256(
                                calibration_result.values[reference],
                                boundary="synthetic reference calibration output",
                            )
                        ),
                        "recipient_output_values_sha256": (
                            stacked_spine_module._post_transfer_float64_sha256(
                                calibration_result.values[recipient],
                                boundary="synthetic recipient calibration output",
                            )
                        ),
                        "reference_weights_sha256": (
                            stacked_spine_module._post_transfer_float64_sha256(
                                weights[reference],
                                boundary="synthetic reference calibration weights",
                            )
                        ),
                        "recipient_weights_sha256": (
                            stacked_spine_module._post_transfer_float64_sha256(
                                weights[recipient],
                                boundary="synthetic recipient calibration weights",
                            )
                        ),
                    },
                },
                "calibration": calibration,
            }
        directions[direction.name] = {
            "targets": target_receipts,
            "post_transfer_calibration": {
                "policy_sha256": policy["sha256"],
                "target_count": len(calibrated_keys),
                "targets": calibrated_keys,
            },
        }
    return {
        "authority": stacked_spine_module.stacked_spine_authority_receipt(),
        "directions": directions,
    }


def _canonical_gap_fill_receipt_with_pattern_evidence() -> tuple[
    dict[str, object],
    str,
    str,
    str,
    str,
    tuple[str, ...],
]:
    receipt = _canonical_gap_fill_calibration_receipt()
    early_keys = {
        spec.key
        for spec in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.stage == "early_gap_fill"
    }
    selected: tuple[str, str, str, str, str, tuple[str, ...]] | None = None
    for direction in stacked_spine_module.CANONICAL_STACKED_GAP_FILL_PLAN:
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                for target in targets:
                    key = f"{entity}/{family}/{target}"
                    if key in early_keys:
                        selected = (
                            direction.name,
                            key,
                            entity,
                            family,
                            target,
                            targets,
                        )
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is not None:
            break
    assert selected is not None
    direction_name, key, entity, family, target, family_targets = selected
    evidence_targets = tuple(
        family_target
        for family_target in family_targets
        if f"{entity}/{family}/{family_target}" in early_keys
    )
    model_targets = acs_transfer_module._model_target_names(evidence_targets)
    required_predictors, optional_predictors = (
        stacked_spine_module._acs_pattern_predictor_authority(
            entity=entity,
            family_targets=family_targets,
        )
    )
    selected_optional = optional_predictors[:1]
    patterns = tuple(
        AcsTransferPattern(
            name=acs_transfer_module._pattern_name(index, observed_optional),
            observed_optional_predictors=observed_optional,
            predictors=(*required_predictors, *observed_optional),
            seed=index,
            weight_kind="design",
            donor_rows=1,
            recipient_rows=1,
            target_regimes=tuple(
                (model_target, "positive_only") for model_target in model_targets
            ),
        )
        for index, observed_optional in enumerate(((), selected_optional))
    )
    record = AcsImputedInput(
        column=target,
        entity=entity,
        family=family,
        donor_spine="synthetic_gap_validator_fixture",
        donor_channel=None,
        predictors=(*required_predictors, *selected_optional),
        seed=0,
        weight_kind="design",
        patterns=patterns,
        imputed_recipient_rows=2,
    )
    target_receipt = receipt["directions"][direction_name]["targets"][key]
    target_receipt.update(
        {
            "authorized_null_rows": 2,
            "imputed_rows": 2,
            "unmodeled_rows": 0,
            "residual_null_rows": 0,
            "qrf_pattern_evidence": (
                stacked_spine_module._acs_imputed_pattern_evidence(record)
            ),
        }
    )
    return receipt, direction_name, key, entity, family, family_targets


def _post_puf_transfer_fixture() -> Frame:
    attached = clone_us_frame_for_puf_support(
        _stacked_gap_fixture(),
        clone_attachment_fraction=0.75,
        clone_attachment_seed=578,
    )
    person = attached.table("person").copy()
    source_producer_rows = (
        person[support_channel_column("person")].astype(str).eq("asec")
    )
    person["is_pregnant"] = pd.Series(
        pd.NA,
        index=person.index,
        dtype="boolean",
    )
    source_eligible = (
        source_producer_rows
        & person["is_female"].astype(bool)
        & person["age"].between(15, 44, inclusive="both")
    )
    person.loc[source_producer_rows, "is_pregnant"] = False
    person.loc[source_eligible, "is_pregnant"] = True
    tables = {entity: attached.table(entity) for entity in attached.entities}
    tables["person"] = person
    return Frame(
        tables,
        attached.schema,
        {entity: attached.weights_for(entity) for entity in attached.weighted_entities},
        attached.strata,
        mass_log=attached.mass_log,
        metadata=attached.metadata,
    )


def _typehugq_cross_origin_readiness_fixture() -> tuple[
    Frame,
    ProducerContract,
    ProducerInput,
]:
    asec_household_count = 1_688
    asec = _source_frame(
        household_ids=list(range(1, asec_household_count + 1)),
        weights=[1.0] * asec_household_count,
        extra_person_columns={"asec_detail_income": 40.0},
        stratum="asec_2024",
    )
    frame = assemble_stacked_spine(
        asec,
        _acs_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame
    primary = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
        stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    ]
    requirement = next(
        item
        for item in primary.inputs
        if item.column == "@effective:validated_structure:TYPEHUGQ"
    )
    contract = replace(primary, inputs=(requirement,), outputs=())
    return frame, contract, requirement


def _fill_late_contract_surface(
    frame: Frame,
    *,
    contracts: tuple[ProducerContract, ...],
    include_outputs: bool,
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    metric_by_column = {
        (entity, column): metric
        for (
            entity,
            _family,
            column,
            _clone_index,
        ), metric in stacked_spine_module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY.items()
    }
    owners = {
        column: entity for entity, table in tables.items() for column in table.columns
    }
    for contract in contracts:
        columns: list[ProducerInputColumn] = []
        for requirement in contract.inputs:
            selected: tuple[ProducerInputColumn, ...] = ()
            for alternative in requirement.alternatives:
                physical = tuple(
                    column
                    for column in alternative
                    if not column.column.startswith("@")
                )
                if not physical and any(
                    column.column != "@resolved_weight" for column in alternative
                ):
                    continue
                if all(
                    column.column not in owners
                    or owners[column.column] == column.entity
                    for column in physical
                ):
                    selected = physical
                    break
            columns.extend(selected)
            owners.update((column.column, column.entity) for column in selected)
        if include_outputs:
            for output in contract.outputs:
                if output.entity == "frame" or output.column.startswith("@"):
                    continue
                columns.append(ProducerInputColumn(output.entity, output.column))
                owners[output.column] = output.entity
        for column in columns:
            table = tables[column.entity]
            if (
                metric_by_column.get((column.entity, column.column))
                == "boolean_incidence"
            ):
                table[column.column] = pd.Series(
                    True,
                    index=table.index,
                    dtype="boolean",
                )
            elif column.column in table:
                table[column.column] = table[column.column].fillna(1)
            elif column.column != "person_support_clone_index":
                table[column.column] = 1.0
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _late_table_digest_vector() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "boolean": pd.array([True, False, pd.NA], dtype="boolean"),
            "integer": pd.array([1, -2, pd.NA], dtype="Int64"),
            "float": np.array([np.inf, -0.0, np.nan], dtype=np.float64),
            "string": pd.array(["", "café", pd.NA], dtype=CANONICAL_STRING_DTYPE),
        },
        index=pd.Index([7, 3, 11], dtype=np.int64, name="row_id"),
    )


@pytest.fixture(autouse=True)
def _prime_worker_identity(prime_primary_qrf_worker_identity: None) -> None:
    """Share the real session attestation unless a test opts into live identity."""


def _fixture_primary_execution_config_binding() -> dict[str, object]:
    return stacked_spine_module._late_primary_execution_config_binding(
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        seed=0,
        n_estimators=100,
        predictors=None,
        person_outputs=None,
        tax_unit_outputs=None,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )


def _validate_fixture_primary_execution_config(
    binding: Mapping[str, object],
) -> None:
    # These cases mutate received bindings, never the installed environment.
    # Reuse the real pristine fixture for comparison, independently of the
    # potentially re-signed candidate; direct identity mutation tests stay live.
    expected_worker = _fixture_primary_execution_config_binding()["qrf"][
        "worker_execution"
    ]
    with pytest.MonkeyPatch.context() as identity_fixture:
        identity_fixture.setattr(
            stacked_spine_module,
            "_late_primary_qrf_worker_execution_binding",
            lambda: deepcopy(expected_worker),
        )
        stacked_spine_module._validate_late_resource_binding(
            binding,
            producer=stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE,
            entity="tax_unit",
            column=stacked_spine_module.US_LATE_PRIMARY_EXECUTION_CONFIG_INPUT,
            boundary="portable worker identity fixture",
        )


def _two_namespace_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    roots = []
    for name in ("installed", "checkout"):
        root = tmp_path / name / "microcosm"
        (root / "build" / "us_runtime").mkdir(parents=True)
        roots.append(root)
    monkeypatch.setattr(
        worker_identity_module.importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=tuple(roots))
            if name == "microcosm"
            else None
        ),
    )
    return roots


def _pyvenv_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str):
    (tmp_path / "pyvenv.cfg").write_text(
        "home = /opt/python/bin\n"
        "implementation = CPython\n"
        "uv = 0.12.9\n"
        f"version_info = {version}\n"
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(worker_identity_module.sys, "prefix", str(tmp_path))


def _stub_worker_identity_static_closure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stub_installed_distributions: bool = True,
) -> None:
    """Keep portable-identity mutation tests focused and inexpensive."""

    monkeypatch.setattr(
        worker_identity_module,
        "_worker_source_identity",
        lambda: ("a" * 64, "b" * 64, ()),
    )
    if stub_installed_distributions:
        monkeypatch.setattr(
            worker_identity_module,
            "_installed_distributions_record_sha256",
            lambda _external_roots: "c" * 64,
        )
    monkeypatch.setattr(
        worker_identity_module,
        "_canonical_pyvenv_config",
        lambda: {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
            "include_system_site_packages": False,
            "uv_version": "fixture",
        },
    )
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "2")


def _fixture_worker_import_trace(
    tmp_path: Path,
    *,
    module_origins: Mapping[str, str] | None = None,
) -> dict[str, object]:
    namespace_root = tmp_path / "namespace" / "microcosm"
    worker_source = namespace_root / "build" / "us_runtime" / "puf_qrf_worker.py"
    worker_source.parent.mkdir(parents=True, exist_ok=True)
    worker_source.write_text("# fixture worker\n", encoding="utf-8")
    return {
        "module_origins": {
            worker_identity_module.PRIMARY_QRF_WORKER_MODULE: str(worker_source),
            **({} if module_origins is None else dict(module_origins)),
        },
        "opened_files": (),
        "namespace_roots": (str(namespace_root),),
    }


@pytest.fixture
def _worker_identity_memo_inputs(
    live_worker_identity, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Exercise the real memo and file hashing with a tiny import closure."""
    _stub_worker_identity_static_closure(monkeypatch)
    runtime = tmp_path / "libpython-fixture.so"
    runtime.write_bytes(b"unchanged fixture runtime\n")
    monkeypatch.setattr(
        worker_identity_module,
        "_loaded_python_runtime_binary",
        lambda: ("shared_library", runtime),
    )
    trace = _fixture_worker_import_trace(tmp_path)
    source = Path(
        trace["module_origins"][worker_identity_module.PRIMARY_QRF_WORKER_MODULE]
    )
    trace["opened_files"] = (str(source),)
    calls = []

    def import_trace():
        calls.append(None)
        return trace

    monkeypatch.setattr(
        worker_identity_module, "_clean_worker_import_trace", import_trace
    )
    return source, calls


def _late_universe_entry_fixture() -> Frame:
    registry = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY
    primary_contract = registry[stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE]
    initial = _fill_late_contract_surface(
        _stacked_gap_fixture(),
        contracts=(primary_contract,),
        include_outputs=False,
    )
    initial_person = initial.table("person")
    structural_row = initial_person.index[
        initial_person[support_channel_column("person")].eq("acs")
    ][0]
    initial_person.loc[structural_row, "age"] = 12.0
    initial_person.loc[
        structural_row,
        [
            "WAGP",
            "SEMP",
            "employment_income_before_lsr",
            "self_employment_income_before_lsr",
        ],
    ] = np.nan
    return initial


def _run_real_late_executor_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bank_identity_sha256: str | None = None,
    bound_clone_attachment_seed: int = 578,
    asec_earnings_delta: float = 0.0,
) -> tuple[stacked_spine_module.StackedLateProducerResult, tuple[str, ...], int]:
    registry = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY
    initial = _late_universe_entry_fixture()
    initial_person = initial.table("person")
    if asec_earnings_delta:
        asec_row = initial_person.index[
            initial_person[support_channel_column("person")].eq("asec")
        ][0]
        initial_person.loc[asec_row, "employment_income_before_lsr"] += (
            asec_earnings_delta
        )
    events: list[str] = []
    finalizer_calls = 0

    materialize_universe = (
        stacked_spine_module._materialize_stacked_acs_earnings_universe
    )

    def universe(
        frame: Frame,
        *,
        execution_config: Mapping[str, object] | None = None,
    ):
        events.append(stacked_spine_module.US_LATE_ACS_EARNINGS_UNIVERSE_STAGE)
        assert execution_config is not None
        return materialize_universe(frame, execution_config=execution_config)

    donor = pd.DataFrame({"fixture_donor": [1.0]})
    actual_primary_resources = (
        stacked_spine_module.stacked_late_primary_resource_receipts(
            donor,
            primary_qrf_checkpoint_identity_sha256="a" * 64,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
            seed=0,
            n_estimators=100,
            fit_records_enabled=True,
            tail_bound_diagnostics_enabled=True,
        )
    )

    def primary(frame: Frame):
        events.append(stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE)
        attached = clone_us_frame_for_puf_support(
            frame,
            clone_attachment_fraction=1.0,
            clone_attachment_seed=578,
        )
        attached = Frame(
            {entity: attached.table(entity) for entity in attached.entities},
            attached.schema,
            {
                entity: attached.weights_for(entity)
                for entity in attached.weighted_entities
            },
            attached.strata,
            mass_log=attached.mass_log,
            metadata={
                **attached.metadata,
                "us_puf_clone_attachment_manifest": {"fixture": True},
            },
        )
        completed = _fill_late_contract_surface(
            attached,
            contracts=tuple(registry.values()),
            include_outputs=True,
        )
        completed_person = completed.table("person")
        completed_person["unemployment_compensation"] = np.ones(
            len(completed_person),
            dtype=np.float64,
        )
        completed_person["is_incapable_of_self_care"] = pd.Series(
            True,
            index=completed_person.index,
            dtype="boolean",
        )
        completed_person["tax_unit_role_input"] = pd.Series(
            "DEPENDENT",
            index=completed_person.index,
            dtype="string",
        )
        adult_recipient = completed_person[support_channel_column("person")].eq(
            "acs"
        ) & completed_person[support_clone_index_column("person")].eq(0)
        completed_person.loc[
            adult_recipient,
            "pre_subsidy_care_expenses",
        ] = 0.0
        adult_carriers = (
            completed_person.loc[adult_recipient]
            .groupby("person_tax_unit_id", sort=False, dropna=False)
            .head(1)
            .index
        )
        completed_person.loc[adult_carriers, "pre_subsidy_care_expenses"] = 1.0
        return stacked_spine_module.StackedPufPassResult(
            completed,
            {
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(actual_primary_resources)
                )
            },
        )

    def source(frame: Frame, operator: str) -> PoolStageOutput:
        events.append(f"source:{operator}")
        return PoolStageOutput(
            frame,
            {
                "phase": "post_clone",
                "operator_order": [operator],
                "suboperators": [{"operator": operator}],
            },
        )

    def finalize(
        frame: Frame,
        *,
        operator_receipts: dict[str, object],
    ) -> PoolStageOutput:
        nonlocal finalizer_calls
        finalizer_calls += 1
        events.append(stacked_spine_module.US_LATE_SOURCE_FINALIZER_STAGE)
        source_order = list(operator_receipts)
        return PoolStageOutput(
            frame,
            {
                "phase": "post_clone",
                "operator_order": source_order,
                "suboperators": [
                    {"operator": operator, "order_index": index}
                    for index, operator in enumerate(source_order)
                ],
                "deferred_transfer_inputs": {
                    "inputs": {
                        column: {}
                        for column in (
                            "bank_account_assets",
                            "bond_assets",
                            "stock_assets",
                        )
                    }
                },
            },
        )

    def transfer(
        frame: Frame,
        *,
        group_name: str,
        execution_contract: Mapping[str, object] | None = None,
        **_kwargs: object,
    ) -> stacked_spine_module.StackedPostPufTransferResult:
        events.append(group_name)
        group = next(
            item
            for item in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
            if item.name == group_name
        )
        assert execution_contract == (
            acs_transfer_module.acs_transfer_execution_contract_identity(
                targets=group.targets,
                derive_schedule_d=False,
            )
        )
        required_predictors, _optional_predictors = (
            stacked_spine_module._acs_pattern_predictor_authority(
                entity=group.entity,
                family_targets=group.targets,
            )
        )
        late_specs = {
            spec.key: spec
            for spec in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
            if spec.stage == "late_transfer"
        }
        evidence_targets = tuple(
            target
            for target in group.targets
            if f"{group.entity}/{group.family}/{target}" in late_specs
        )
        model_targets = acs_transfer_module._model_target_names(evidence_targets)
        pattern = AcsTransferPattern(
            name="pattern_00_e3b0c442",
            observed_optional_predictors=(),
            predictors=required_predictors,
            seed=0,
            weight_kind="design",
            donor_rows=1,
            recipient_rows=1,
            target_regimes=tuple((target, "positive_only") for target in model_targets),
        )
        plain_pattern = replace(pattern, target_regimes=())
        synthetic_imputed_inputs = tuple(
            AcsImputedInput(
                column=target,
                entity=group.entity,
                family=group.family,
                donor_spine="synthetic_late_executor_fixture",
                donor_channel="asec",
                predictors=pattern.predictors,
                seed=pattern.seed,
                weight_kind=pattern.weight_kind,
                patterns=(pattern if target in evidence_targets else plain_pattern,),
                imputed_recipient_rows=1,
            )
            for target in group.targets
        )
        transfer_result = AcsTransferResult(
            frame=frame,
            imputed_inputs=synthetic_imputed_inputs,
            fit_records=(),
            deferred_inputs=(),
            resolved_donor_channel="asec",
        )
        policy_sha256 = post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()[
            "sha256"
        ]
        target_receipts: dict[str, dict[str, object]] = {}
        for target, record in zip(
            group.targets,
            synthetic_imputed_inputs,
            strict=True,
        ):
            key = f"{group.entity}/{group.family}/{target}"
            target_receipt: dict[str, object] = {
                "authorized_null_rows": 1,
                "imputed_rows": 1,
                "unmodeled_rows": 0,
                "residual_null_rows": 0,
            }
            if target == "is_pregnant":
                pregnancy_policy = execution_contract["structural_target_policies"][
                    "is_pregnant"
                ]
                target_receipt["structural_policy"] = {
                    "policy_sha256": pregnancy_policy["sha256"],
                    "source_person_key": "person_source_id",
                    "source_persons_checked": 1,
                    "physical_rows_checked": 1,
                    "clone_rows_checked": 0,
                    "donor_rows_checked": 1,
                    "qrf_draw_source_persons": 1,
                    "qrf_draw_rows": 1,
                    "qrf_fanout_rows": 0,
                    "preexisting_value_fanout_rows": 0,
                    "ineligible_rows_assigned_false": 0,
                    "donor_preexisting_domain_violation_rows": 0,
                    "recipient_preexisting_domain_violation_rows": 0,
                    "preexisting_clone_disagreement_source_persons": 0,
                    "inconsistent_eligibility_source_persons": 0,
                    "maximum_clones_per_source_person": 1,
                    "final_incomplete_rows": 0,
                    "final_domain_violation_rows": 0,
                    "final_clone_disagreement_source_persons": 0,
                    "status": "verified",
                }
            if key in late_specs:
                target_receipt["qrf_pattern_evidence"] = (
                    stacked_spine_module._acs_imputed_pattern_evidence(record)
                )
            target_receipts[key] = target_receipt
        calibrated_keys = sorted(set(target_receipts) & set(late_specs))
        for key in calibrated_keys:
            spec = late_specs[key]
            constrained = spec.special_constraint != "none"
            live_table = frame.table(spec.entity)
            live_channel = live_table[support_channel_column(spec.entity)].astype(str)
            live_clone = pd.to_numeric(
                live_table[support_clone_index_column(spec.entity)],
                errors="raise",
            )
            live_reference = (live_channel.eq("asec") & live_clone.eq(0)).to_numpy(
                dtype=bool
            )
            live_recipient = (live_channel.eq("acs") & live_clone.eq(0)).to_numpy(
                dtype=bool
            )
            allowed_rows: np.ndarray | None = None
            addition_rows: np.ndarray | None = None
            if spec.special_constraint == ("adult_care_qualifying_one_per_tax_unit"):
                mutable_series = pd.Series(
                    live_recipient,
                    index=live_table.index,
                    dtype=bool,
                )
                allowed_series = (
                    mutable_series
                    & acs_transfer_module.acs_adult_care_qualifying_rows(live_table)
                )
                addition_series = (
                    stacked_spine_module._one_candidate_per_adult_care_tax_unit(
                        frame,
                        mutable_rows=mutable_series,
                        allowed_rows=allowed_series,
                    )
                )
                allowed_rows = allowed_series.to_numpy(dtype=bool)
                addition_rows = addition_series.to_numpy(dtype=bool)
            elif spec.special_constraint == (
                "weeks_requires_positive_unemployment_compensation"
            ):
                allowed_rows = live_recipient & pd.to_numeric(
                    live_table["unemployment_compensation"],
                    errors="raise",
                ).gt(0.0).to_numpy(dtype=bool)
                addition_rows = allowed_rows.copy()
            application = (
                post_transfer_calibration_runtime.apply_post_transfer_calibration(
                    frame,
                    entity=spec.entity,
                    family=spec.family,
                    target=spec.target,
                    reference_rows=live_reference,
                    recipient_rows=live_recipient,
                    mutable_rows=live_recipient,
                    allowed_carrier_rows=allowed_rows if constrained else None,
                    addition_candidate_rows=addition_rows if constrained else None,
                )
            )
            frame = application.frame
            calibration = application.receipt
            scope = calibration["scope"]
            constraint: dict[str, object] = {"constraint": spec.special_constraint}
            if spec.special_constraint == ("adult_care_qualifying_one_per_tax_unit"):
                constraint.update(
                    {
                        "qualifying_mutable_rows": scope["allowed_carrier_rows"],
                        "one_per_empty_tax_unit_addition_candidates": scope[
                            "addition_candidate_rows"
                        ],
                    }
                )
            elif spec.special_constraint == (
                "weeks_requires_positive_unemployment_compensation"
            ):
                constraint["positive_unemployment_mutable_rows"] = scope[
                    "allowed_carrier_rows"
                ]
            owner: dict[str, object] = {
                "stage": "late_transfer",
                "reference_selection": "asec_origin_clone_0",
                "recipient_selection": "acs_origin_clone_0",
                "mutable_selection": "recipient_null_before_nonnull_after",
                "reference_rows": scope["reference_rows"],
                "recipient_rows": scope["recipient_rows"],
                "mutable_rows": scope["mutable_rows"],
                "constraint": constraint,
                "context_binding": {
                    "scope": dict(scope),
                    "weights_sha256": calibration["weights"]["sha256"],
                    "live_output": (
                        stacked_spine_module._post_transfer_selected_output_binding(
                            frame,
                            entity=spec.entity,
                            target=spec.target,
                            reference_rows=live_reference,
                            recipient_rows=live_recipient,
                        )
                    ),
                },
                "calibration": calibration,
            }
            if spec.special_constraint == ("adult_care_qualifying_one_per_tax_unit"):
                owner["post_reconciliation"] = {"status": "verified_no_op"}
            target_receipts[key]["post_transfer_calibration"] = owner
        return stacked_spine_module.StackedPostPufTransferResult(
            frame,
            {
                "producer": group.name,
                "ordered_targets": list(group.targets),
                "targets": target_receipts,
                "post_transfer_calibration": {
                    "policy_sha256": policy_sha256,
                    "target_count": len(calibrated_keys),
                    "targets": calibrated_keys,
                },
            },
            replace(transfer_result, frame=frame),
        )

    monkeypatch.setattr(
        stacked_spine_module,
        "_materialize_stacked_acs_earnings_universe",
        universe,
    )
    monkeypatch.setattr(
        multispine_pool_module,
        "run_multispine_post_clone_source_operator",
        source,
    )
    monkeypatch.setattr(
        multispine_pool_module,
        "finalize_multispine_source_inputs",
        finalize,
    )
    monkeypatch.setattr(
        stacked_spine_module,
        "transfer_stacked_post_puf_group",
        transfer,
    )
    resources = stacked_spine_module.stacked_late_primary_resource_receipts(
        donor,
        primary_qrf_checkpoint_identity_sha256="a" * 64,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=bound_clone_attachment_seed,
        seed=0,
        n_estimators=100,
        fit_records_enabled=True,
        tail_bound_diagnostics_enabled=True,
    )
    target_banks = None
    if bank_identity_sha256 is not None:

        class IdentityBank:
            identity_sha256 = bank_identity_sha256

        target_banks = {
            group.name: IdentityBank()
            for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
        }

    result = stacked_spine_module.run_stacked_late_producer_dag(
        initial,
        primary_puf_producer=primary,
        primary_resource_receipts=resources,
        target_banks=target_banks,
    )

    return result, tuple(events), finalizer_calls


def _rehash_late_receipt_after_fixture_mutation(
    receipt: dict[str, object],
) -> None:
    previous = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=receipt["producer_schedule"]["payload_sha256"],
        input_frame_sha256=receipt["input_frame_sha256"],
    )
    for row in receipt["execution"]:
        row["input_surface_sha256"] = stacked_spine_module._canonical_sha256(
            row["declared_inputs"]
        )
        row["previous_execution_sha256"] = previous
        row.pop("sha256", None)
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous = row["sha256"]
    receipt["execution_chain_sha256"] = previous
    receipt.pop("sha256", None)
    receipt["sha256"] = stacked_spine_module._canonical_sha256(receipt)


def _post_puf_puf_producer_fixture() -> Frame:
    frame = _post_puf_transfer_fixture()
    person = frame.table("person").copy()
    clone_index = pd.to_numeric(
        person[support_clone_index_column("person")],
        errors="raise",
    )
    producer_rows = clone_index.gt(0)
    person["educator_expense"] = np.nan
    person.loc[producer_rows, "educator_expense"] = 10.0 * np.arange(
        1,
        int(producer_rows.sum()) + 1,
    )
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _completed_stacked_frame() -> Frame:
    """A stacked fixture whose declared surface is fully observed."""

    return assemble_stacked_spine(
        _asec_gap_source(),
        _acs_gap_source(),
        acs_sample_fraction=1.0,
        acs_sample_seed=578,
    ).frame


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


def _battery_frame(
    columns: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    acs_group_quarters: bool = False,
) -> Frame:
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
        if position == 1 and acs_group_quarters:
            person.loc[person.index[0], "pre_subsidy_rent"] = np.nan
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = person
        household = tables["household"].copy()
        household["TYPEHUGQ"] = np.ones(len(household), dtype=np.int64)
        household["tenure_type"] = "RENTED"
        if position == 1 and acs_group_quarters:
            household.loc[household.index[0], "TYPEHUGQ"] = 2
            household.loc[household.index[0], "tenure_type"] = np.nan
        tables["household"] = household
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
    person["self_employment_income_before_lsr"] = 0.0
    # Structurally learnable from a REQUIRED predictor with a stable share
    # under any household subsample, so the gap-fill QRF reproduces both
    # incidences on the seeded ACS sample without sampling-skew noise.
    person["unemployment_compensation"] = np.where(person["is_female"], 2_400.0, 0.0)
    person["is_disabled"] = person["is_female"].to_numpy()
    person["pre_subsidy_rent"] = np.where(index % 2 == 1, 11_000.0 + 150.0 * index, 0.0)
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
    household["TYPEHUGQ"] = np.ones(40, dtype=np.int64)
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
    person["WAGP"] = person["employment_income_before_lsr"]
    person["self_employment_income_before_lsr"] = 0.0
    person["SEMP"] = 0.0
    person["acs_interest_dividend_rental_income"] = np.where(
        index % 2 == 0, 1_250.0 + 42.0 * index, 0.0
    )
    household = frame.table("household").copy()
    household["TYPEHUGQ"] = np.ones(40, dtype=np.int64)
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
        name="asec_housing_to_acs",
        recipient_channel="acs",
        donor_channel="asec",
        target_families={"person": {"housing": ("pre_subsidy_rent",)}},
        recipient_absence_rules=(_CANONICAL_RENT_ABSENCE_RULE,),
    ),
)

__all__ = [name for name in globals() if not name.startswith("__")]
