"""US source operation handlers for declarative manifests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from populace.build.source_manifest import SourceOperationSpec
from populace.build.source_runtime import (
    SourceOperationHandler,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from populace.build.us_runtime.capital_gain_distributions import (
    split_us_component_by_share_from_manifest,
)
from populace.build.us_runtime.childcare import (
    derive_us_childcare_from_manifest,
    impute_us_childcare_to_puf_support_from_manifest,
)
from populace.build.us_runtime.education_inputs import (
    derive_us_education_inputs_from_manifest,
)
from populace.build.us_runtime.eligibility_inputs import (
    derive_us_eligibility_inputs_from_manifest,
)
from populace.build.us_runtime.hours_worked import (
    derive_us_hours_worked_from_manifest,
)
from populace.build.us_runtime.immigration import (
    derive_us_immigration_status_from_manifest,
)
from populace.build.us_runtime.pregnancy import (
    derive_us_pregnancy_from_manifest,
)
from populace.build.us_runtime.puf_aggregate_records import (
    derive_puf_policyengine_variables,
    disaggregate_puf_aggregate_records,
    load_default_puf_aggregate_disaggregation_spec,
)
from populace.build.us_runtime.retirement_contributions import (
    derive_us_retirement_contributions_from_manifest,
    impute_us_retirement_contributions_to_puf_support_from_manifest,
)
from populace.build.us_runtime.snap_discretionary_exemption import (
    derive_us_snap_discretionary_exemption_from_manifest,
)
from populace.build.us_runtime.snap_take_up import (
    derive_us_snap_take_up_from_manifest,
)

__all__ = [
    "aggregate_us_person_to_tax_unit_from_manifest",
    "assign_us_binary_from_rate_from_manifest",
    "calibrate_us_binary_assignment_from_manifest",
    "calibrate_us_binary_assignment_joint_targets_from_manifest",
    "compute_us_ratio_from_manifest",
    "derive_us_childcare_from_manifest",
    "derive_us_puf_policyengine_variables_from_manifest",
    "derive_us_retirement_contributions_from_manifest",
    "disaggregate_us_puf_aggregate_records_from_manifest",
    "impute_us_childcare_to_puf_support_from_manifest",
    "impute_us_retirement_contributions_to_puf_support_from_manifest",
    "support_clip_us_source_output_from_manifest",
    "us_source_operation_handlers",
]

_PUF_AGGREGATE_DISAGGREGATION_PARAMETER_KEYS = frozenset(
    {
        "method",
        "spec",
        "replace_records",
        "weight",
        "amount_columns",
        "seed_from_build_config",
    }
)

_PUF_POLICYENGINE_VARIABLE_PARAMETER_KEYS = frozenset(
    {
        "ordinary_dividend_source",
        "qualified_dividend_source",
        "qualified_dividend_output",
        "non_qualified_dividend_output",
        "qualified_tuition_primary_source",
        "qualified_tuition_optional_source",
        "qualified_tuition_output",
        "casualty_loss_source",
        "casualty_loss_output",
        "unreimbursed_business_employee_expenses_source",
        "unreimbursed_business_employee_expenses_output",
    }
)

_ACA_AGGREGATE_PERSON_TO_TAX_UNIT_PARAMETER_KEYS = frozenset(
    {
        "person_table",
        "tax_unit_table",
        "person_tax_unit_id",
        "tax_unit_id",
        "aggregates",
        "operation",
    }
)

_ASSIGN_BINARY_FROM_RATE_PARAMETER_KEYS = frozenset(
    {
        "output",
        "draw",
        "rate_key",
        "rate_column",
        "eligibility",
        "reported_true_anchor",
    }
)

_CALIBRATE_BINARY_ASSIGNMENT_PARAMETER_KEYS = frozenset(
    {
        "variable",
        "targets",
        "optional",
        "preserve_true_anchors",
        "preserve_true_anchor",
        "domain",
        "group_by",
        "weight",
        "draw",
    }
)

_CALIBRATE_BINARY_JOINT_TARGET_PARAMETER_KEYS = frozenset(
    {
        "variable",
        "count_targets",
        "amount_targets",
        "optional",
        "preserve_true_anchors",
        "preserve_true_anchor",
        "domain",
        "group_by",
        "count_weight",
        "amount_weight",
        "draw",
    }
)

_ACA_COMPUTE_RATIO_PARAMETER_KEYS = frozenset(
    {
        "output",
        "numerator",
        "denominator",
        "where",
    }
)

_SOURCE_SUPPORT_CLIP_PARAMETER_KEYS = frozenset(
    {
        "output",
        "lower",
        "upper",
        "range",
    }
)

_TARGET_VALUE_COLUMNS = ("target", "value", "count", "n")


@dataclass(frozen=True)
class _BinaryCalibrationValueState:
    """A boolean calibration target backed by a column or simple comparison."""

    column: str
    less_than_threshold: float | None = None

    def values(self, frame: pd.DataFrame) -> pd.Series:
        if self.less_than_threshold is None:
            return frame[self.column].fillna(False).astype(bool)
        numeric = pd.to_numeric(frame[self.column], errors="coerce").fillna(
            self.less_than_threshold
        )
        return numeric < self.less_than_threshold

    def write(
        self,
        frame: pd.DataFrame,
        values: pd.Series,
        domain: pd.Series,
    ) -> pd.DataFrame:
        result = frame.copy(deep=True)
        if self.less_than_threshold is None:
            result[self.column] = values.astype(bool)
            return result

        threshold = self.less_than_threshold
        epsilon = 1e-6
        numeric = pd.to_numeric(result[self.column], errors="coerce").fillna(threshold)
        make_true = values & domain
        make_false = (~values) & domain
        numeric.loc[make_true & ~(numeric < threshold)] = threshold - epsilon
        numeric.loc[make_false & (numeric < threshold)] = threshold
        result[self.column] = numeric
        return result


def us_source_operation_handlers() -> Mapping[str, SourceOperationHandler]:
    """Return US handlers keyed by manifest operation kind."""

    return {
        "aggregate_person_to_tax_unit": aggregate_us_person_to_tax_unit_from_manifest,
        "assign_binary_from_rate": assign_us_binary_from_rate_from_manifest,
        "calibrate_binary_assignment": (calibrate_us_binary_assignment_from_manifest),
        "calibrate_binary_assignment_joint_targets": (
            calibrate_us_binary_assignment_joint_targets_from_manifest
        ),
        "compute_ratio": compute_us_ratio_from_manifest,
        "derive_childcare_inputs": derive_us_childcare_from_manifest,
        "derive_eligibility_inputs": derive_us_eligibility_inputs_from_manifest,
        "derive_education_inputs": derive_us_education_inputs_from_manifest,
        "derive_hours_worked": derive_us_hours_worked_from_manifest,
        "derive_pregnancy": derive_us_pregnancy_from_manifest,
        "derive_retirement_contributions": (
            derive_us_retirement_contributions_from_manifest
        ),
        "impute_childcare_to_puf_support": (
            impute_us_childcare_to_puf_support_from_manifest
        ),
        "impute_retirement_contributions_to_puf_support": (
            impute_us_retirement_contributions_to_puf_support_from_manifest
        ),
        "derive_snap_abawd_discretionary_exemption": derive_us_snap_discretionary_exemption_from_manifest,
        "derive_immigration_status": derive_us_immigration_status_from_manifest,
        "derive_snap_take_up": derive_us_snap_take_up_from_manifest,
        "derive_puf_policyengine_variables": (
            derive_us_puf_policyengine_variables_from_manifest
        ),
        "disaggregate_aggregate_records": (
            disaggregate_us_puf_aggregate_records_from_manifest
        ),
        "split_component_by_share": split_us_component_by_share_from_manifest,
        "support_clip": support_clip_us_source_output_from_manifest,
    }


def aggregate_us_person_to_tax_unit_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Aggregate CPS person Marketplace coverage fields to tax units."""

    if operation.kind != "aggregate_person_to_tax_unit":
        raise SourceRuntimeError(
            "ACA tax-unit aggregation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is not None:
        raise SourceRuntimeError(
            "ACA tax-unit aggregation must be the first operation in the stage."
        )
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_ACA_AGGREGATE_PERSON_TO_TAX_UNIT_PARAMETER_KEYS,
        label="ACA tax-unit aggregation",
    )
    if params.get("operation") != "any":
        raise SourceRuntimeError(
            "ACA tax-unit aggregation currently supports only operation='any'."
        )
    person_table = _required_string_param(
        params,
        "person_table",
        label="ACA tax-unit aggregation",
    )
    tax_unit_table = _required_string_param(
        params,
        "tax_unit_table",
        label="ACA tax-unit aggregation",
    )
    person_tax_unit_id = _required_string_param(
        params,
        "person_tax_unit_id",
        label="ACA tax-unit aggregation",
    )
    tax_unit_id = _required_string_param(
        params,
        "tax_unit_id",
        label="ACA tax-unit aggregation",
    )
    aggregates = _required_string_sequence_param(
        params,
        "aggregates",
        label="ACA tax-unit aggregation",
    )

    people = context.read_table(person_table)
    tax_units = context.read_table(tax_unit_table)
    _require_columns(
        people,
        [person_tax_unit_id, *aggregates],
        label=person_table,
    )
    _require_columns(tax_units, [tax_unit_id], label=tax_unit_table)

    grouped = (
        people.assign(
            **{name: people[name].fillna(False).astype(bool) for name in aggregates}
        )
        .groupby(person_tax_unit_id, sort=False)[list(aggregates)]
        .any()
        .reset_index()
        .rename(columns={person_tax_unit_id: tax_unit_id})
    )
    result = tax_units.drop(columns=[c for c in aggregates if c in tax_units])
    result = result.merge(grouped, on=tax_unit_id, how="left")
    for name in aggregates:
        result[name] = result[name].fillna(False).astype(bool)
    return result


def assign_us_binary_from_rate_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Assign a seeded binary take-up flag from rates and reported anchors."""

    label = "US binary assignment"
    if operation.kind != "assign_binary_from_rate":
        raise SourceRuntimeError(
            f"{label} received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(f"{label} requires a current frame.")
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_ASSIGN_BINARY_FROM_RATE_PARAMETER_KEYS,
        label=label,
    )
    output = _required_string_param(params, "output", label=label)
    draw_column = _required_string_param(params, "draw", label=label)
    rate_key = _required_string_param(params, "rate_key", label=label)
    rate_column = params.get("rate_column", f"{rate_key}_take_up_rate")
    if not isinstance(rate_column, str) or not rate_column:
        raise SourceRuntimeError(f"{label} requires a rate column.")
    _require_columns(frame, [rate_column], label=f"{label} frame")

    result = frame.copy(deep=True)
    if draw_column in result:
        draws = pd.to_numeric(result[draw_column], errors="coerce")
        if draws.isna().any():
            raise SourceRuntimeError(
                f"{label} draw column {draw_column!r} contains nulls."
            )
        draws = draws.clip(lower=0.0, upper=1.0)
    else:
        # The "aca:" salt prefix predates generic reuse of this handler and is
        # frozen: existing ACA releases are bit-reproducible against it.
        # Stages at other grains should supply the draw column themselves.
        draws = _stable_draws(
            result,
            seed=int(context.config.seed),
            salt=f"aca:{output}",
        )
        result[draw_column] = draws

    rates = pd.to_numeric(result[rate_column], errors="coerce").fillna(0.0)
    assigned = draws < rates.clip(lower=0.0, upper=1.0)

    eligibility = params.get("eligibility")
    if isinstance(eligibility, str) and eligibility:
        _require_columns(result, [eligibility], label=f"{label} frame")
        eligibility_mask = result[eligibility].fillna(False).astype(bool)
        assigned = assigned & eligibility_mask
    else:
        eligibility_mask = pd.Series(True, index=result.index)

    anchor = params.get("reported_true_anchor")
    if isinstance(anchor, str) and anchor:
        _require_columns(result, [anchor], label=f"{label} frame")
        assigned = assigned | (
            result[anchor].fillna(False).astype(bool) & eligibility_mask
        )

    result[output] = assigned.astype(bool)
    return result


def calibrate_us_binary_assignment_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Greedily calibrate a boolean assignment to count targets."""

    label = "US binary calibration"
    if operation.kind != "calibrate_binary_assignment":
        raise SourceRuntimeError(
            f"{label} received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(f"{label} requires a current frame.")
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_CALIBRATE_BINARY_ASSIGNMENT_PARAMETER_KEYS,
        label=label,
    )
    variable = _required_string_param(
        params,
        "variable",
        label=label,
    )
    targets = _required_string_sequence_param(
        params,
        "targets",
        label=label,
    )
    value_state = _binary_calibration_value_state(frame, variable, label=label)

    try:
        target_table_name, target_table = _first_available_target_table(
            targets, context, label=label
        )
    except SourceRuntimeError:
        if bool(params.get("optional", False)):
            return frame
        raise
    value_column = _target_value_column(target_table, label=target_table_name)
    group_by = _group_columns(params.get("group_by"), target_table, frame, label=label)
    weight_column = params.get("weight")
    draw_column = _required_string_param(params, "draw", label=label)
    if not isinstance(weight_column, str) or not weight_column:
        raise SourceRuntimeError(f"{label} requires an explicit weight column.")

    result = frame.copy(deep=True)
    _require_columns(result, [weight_column], label=f"{label} frame")
    weights = pd.to_numeric(result[weight_column], errors="coerce").fillna(0.0)
    if draw_column not in result:
        result[draw_column] = _stable_draws(
            result,
            seed=int(context.config.seed),
            salt=f"calibrate:{variable}",
        )
    draws = pd.to_numeric(result[draw_column], errors="coerce").fillna(1.0)

    values = value_state.values(result)
    preserved = _preserved_true_mask(result, params, label=label)
    domain = _domain_mask(result, params.get("domain"), label=label)
    candidate_mask = (~preserved) & domain

    target_rows = target_table.copy(deep=True)
    if not group_by:
        target_rows = target_rows.assign(__all__="all")
        result = result.assign(__all__="all")
        group_by = ("__all__",)

    _require_columns(target_rows, [*group_by, value_column], label=target_table_name)
    _require_columns(result, group_by, label=f"{label} frame")

    calibrated = values.copy()
    for _, target_row in target_rows.iterrows():
        group_mask = pd.Series(True, index=result.index)
        for column in group_by:
            group_mask &= result[column] == target_row[column]
        target_value = float(target_row[value_column])
        active_group = group_mask & domain
        existing = float(weights[active_group & calibrated].sum())
        remaining = target_value - existing
        if remaining > 0:
            eligible = result.index[active_group & candidate_mask & (~calibrated)]
            ordered = sorted(
                eligible, key=lambda idx: (float(draws.loc[idx]), str(idx))
            )
            added = 0.0
            for idx in ordered:
                if added >= remaining:
                    break
                calibrated.loc[idx] = True
                added += float(weights.loc[idx])
            continue

        excess = -remaining
        if excess <= 0:
            continue
        eligible = result.index[active_group & candidate_mask & calibrated]
        ordered = sorted(eligible, key=lambda idx: (-float(draws.loc[idx]), str(idx)))
        removed = 0.0
        for idx in ordered:
            if removed >= excess:
                break
            calibrated.loc[idx] = False
            removed += float(weights.loc[idx])

    result = value_state.write(result, calibrated, domain)
    if "__all__" in result.columns:
        result = result.drop(columns=["__all__"])
    return result


def calibrate_us_binary_assignment_joint_targets_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Calibrate a boolean assignment to linked count and amount targets.

    For each target group, this starts from preserved true anchors, then fills
    the remaining count target with units whose amount per counted unit is
    closest to the target amount/count average. This is intentionally read-only
    with respect to target values: the manifest supplies target tables and
    source columns, while the runtime supplies the generic selection rule.
    """

    label = "US joint binary calibration"
    if operation.kind != "calibrate_binary_assignment_joint_targets":
        raise SourceRuntimeError(
            f"{label} received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(f"{label} requires a frame.")
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_CALIBRATE_BINARY_JOINT_TARGET_PARAMETER_KEYS,
        label=label,
    )
    variable = _required_string_param(
        params,
        "variable",
        label=label,
    )
    count_targets = _required_string_sequence_param(
        params,
        "count_targets",
        label=label,
    )
    amount_targets = _required_string_sequence_param(
        params,
        "amount_targets",
        label=label,
    )
    try:
        count_target_name, count_target_table = _first_available_target_table(
            count_targets, context, label=label
        )
        amount_target_name, amount_target_table = _first_available_target_table(
            amount_targets, context, label=label
        )
    except SourceRuntimeError:
        if bool(params.get("optional", False)):
            return frame
        raise

    count_value_column = _target_value_column(
        count_target_table,
        label=count_target_name,
    )
    amount_value_column = _target_value_column(
        amount_target_table,
        label=amount_target_name,
    )
    group_by = _group_columns(
        params.get("group_by"), count_target_table, frame, label=label
    )
    count_weight_column = params.get("count_weight")
    amount_weight_column = params.get("amount_weight")
    draw_column = _required_string_param(params, "draw", label=label)
    if not isinstance(count_weight_column, str) or not count_weight_column:
        raise SourceRuntimeError(f"{label} requires a count_weight column.")
    if not isinstance(amount_weight_column, str) or not amount_weight_column:
        raise SourceRuntimeError(f"{label} requires an amount_weight column.")

    result = frame.copy(deep=True)
    if not group_by:
        count_target_table = count_target_table.assign(__all__="all")
        amount_target_table = amount_target_table.assign(__all__="all")
        result = result.assign(__all__="all")
        group_by = ("__all__",)

    _require_columns(
        count_target_table,
        [*group_by, count_value_column],
        label=count_target_name,
    )
    _require_columns(
        amount_target_table,
        [*group_by, amount_value_column],
        label=amount_target_name,
    )
    _require_columns(
        result,
        [*group_by, count_weight_column, amount_weight_column],
        label=f"{label} frame",
    )
    if draw_column not in result:
        result[draw_column] = _stable_draws(
            result,
            seed=int(context.config.seed),
            salt=f"joint-calibrate:{variable}",
        )

    value_state = _binary_calibration_value_state(result, variable, label=label)
    values = value_state.values(result)
    preserved = _preserved_true_mask(result, params, label=label)
    domain = _domain_mask(result, params.get("domain"), label=label)
    count_weights = pd.to_numeric(
        result[count_weight_column],
        errors="coerce",
    ).fillna(0.0)
    amount_weights = pd.to_numeric(
        result[amount_weight_column],
        errors="coerce",
    ).fillna(0.0)
    draws = pd.to_numeric(result[draw_column], errors="coerce").fillna(1.0)
    unit_amount = amount_weights / count_weights.where(count_weights > 0)
    unit_amount = unit_amount.fillna(0.0)

    amount_by_key = {
        tuple(row[column] for column in group_by): float(row[amount_value_column])
        for _, row in amount_target_table.iterrows()
    }
    calibrated = values.copy()
    for _, target_row in count_target_table.iterrows():
        key = tuple(target_row[column] for column in group_by)
        if key not in amount_by_key:
            raise SourceRuntimeError(
                f"{amount_target_name} is missing group "
                f"{dict(zip(group_by, key, strict=True))}."
            )
        group_mask = pd.Series(True, index=result.index)
        for column, value in zip(group_by, key, strict=True):
            group_mask &= result[column] == value

        active_group = group_mask & domain
        calibrated.loc[active_group] = False
        preserved_group = active_group & preserved
        calibrated.loc[preserved_group] = True

        target_count = float(target_row[count_value_column])
        target_amount = amount_by_key[key]
        preserved_count = float(count_weights[preserved_group].sum())
        preserved_amount = float(amount_weights[preserved_group].sum())
        remaining_count = target_count - preserved_count
        if remaining_count <= 0:
            continue
        remaining_amount = max(target_amount - preserved_amount, 0.0)
        target_average = remaining_amount / remaining_count

        eligible = result.index[
            active_group & (~preserved) & (count_weights > 0) & (amount_weights >= 0)
        ]
        ordered = sorted(
            eligible,
            key=lambda idx: (
                abs(float(unit_amount.loc[idx]) - target_average),
                float(draws.loc[idx]),
                str(idx),
            ),
        )
        added = 0.0
        for idx in ordered:
            if added >= remaining_count:
                break
            calibrated.loc[idx] = True
            added += float(count_weights.loc[idx])

    result = value_state.write(result, calibrated, domain)
    if "__all__" in result.columns:
        result = result.drop(columns=["__all__"])
    return result


def compute_us_ratio_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Compute a declared source ratio with neutral defaults off-domain."""

    if operation.kind != "compute_ratio":
        raise SourceRuntimeError(
            f"US ratio computation received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError("US ratio computation requires a current frame.")
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_ACA_COMPUTE_RATIO_PARAMETER_KEYS,
        label="US ratio computation",
    )
    output = _required_string_param(params, "output", label="US ratio computation")
    denominator = _required_string_param(
        params,
        "denominator",
        label="US ratio computation",
    )
    numerator = _required_string_sequence_param(
        params,
        "numerator",
        label="US ratio computation",
    )
    where = params.get("where")
    if not isinstance(where, str) or not where:
        raise SourceRuntimeError(
            "US ratio computation requires a boolean where column."
        )

    _require_columns(frame, [*numerator, denominator, where], label="ratio frame")
    result = frame.copy(deep=True)
    numerator_values = sum(
        pd.to_numeric(result[column], errors="coerce").fillna(0.0)
        for column in numerator
    )
    denominator_values = pd.to_numeric(result[denominator], errors="coerce").fillna(0.0)
    domain = result[where].fillna(False).astype(bool) & (denominator_values > 0.0)
    ratio = pd.Series(1.0, index=result.index, dtype=float)
    ratio.loc[domain] = numerator_values.loc[domain] / denominator_values.loc[domain]
    result[output] = ratio
    return result


def derive_us_puf_policyengine_variables_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Translate raw IRS PUF columns into PE-aligned source variables."""

    _ = context
    if operation.kind != "derive_puf_policyengine_variables":
        raise SourceRuntimeError(
            "PUF PolicyEngine-variable derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "PUF PolicyEngine-variable derivation requires a current source frame."
        )
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_PUF_POLICYENGINE_VARIABLE_PARAMETER_KEYS,
        label="PUF PolicyEngine-variable derivation",
    )
    try:
        return derive_puf_policyengine_variables(
            frame,
            ordinary_dividend_source=_string_param_with_default(
                params,
                "ordinary_dividend_source",
                default="E00600",
                label="PUF PolicyEngine-variable derivation",
            ),
            qualified_dividend_source=_string_param_with_default(
                params,
                "qualified_dividend_source",
                default="E00650",
                label="PUF PolicyEngine-variable derivation",
            ),
            qualified_dividend_output=_string_param_with_default(
                params,
                "qualified_dividend_output",
                default="qualified_dividend_income",
                label="PUF PolicyEngine-variable derivation",
            ),
            non_qualified_dividend_output=_string_param_with_default(
                params,
                "non_qualified_dividend_output",
                default="non_qualified_dividend_income",
                label="PUF PolicyEngine-variable derivation",
            ),
            qualified_tuition_primary_source=_optional_string_param(
                params,
                "qualified_tuition_primary_source",
                label="PUF PolicyEngine-variable derivation",
            ),
            qualified_tuition_optional_source=_optional_string_param(
                params,
                "qualified_tuition_optional_source",
                label="PUF PolicyEngine-variable derivation",
            ),
            qualified_tuition_output=_string_param_with_default(
                params,
                "qualified_tuition_output",
                default="qualified_tuition_expenses",
                label="PUF PolicyEngine-variable derivation",
            ),
            casualty_loss_source=_optional_string_param(
                params,
                "casualty_loss_source",
                label="PUF PolicyEngine-variable derivation",
            ),
            casualty_loss_output=_string_param_with_default(
                params,
                "casualty_loss_output",
                default="casualty_loss",
                label="PUF PolicyEngine-variable derivation",
            ),
            unreimbursed_business_employee_expenses_source=_optional_string_param(
                params,
                "unreimbursed_business_employee_expenses_source",
                label="PUF PolicyEngine-variable derivation",
            ),
            unreimbursed_business_employee_expenses_output=(
                _string_param_with_default(
                    params,
                    "unreimbursed_business_employee_expenses_output",
                    default="unreimbursed_business_employee_expenses",
                    label="PUF PolicyEngine-variable derivation",
                )
            ),
        )
    except ValueError as exc:
        raise SourceRuntimeError(str(exc)) from exc


def support_clip_us_source_output_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Clip a declared source output to a reviewed support interval."""

    if operation.kind != "support_clip":
        raise SourceRuntimeError(
            f"US support clip received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError("US support clip requires a current frame.")
    params = operation.parameters
    _reject_unexpected_parameters(
        params,
        allowed=_SOURCE_SUPPORT_CLIP_PARAMETER_KEYS,
        label="US support clip",
    )
    output = _required_string_param(params, "output", label="US support clip")
    lower = params.get("lower")
    upper = params.get("upper")
    if not isinstance(lower, int | float) or not isinstance(upper, int | float):
        raise SourceRuntimeError("US support clip requires numeric lower and upper.")
    if lower > upper:
        raise SourceRuntimeError("US support clip lower bound exceeds upper bound.")
    _require_columns(frame, [output], label="support-clip frame")

    result = frame.copy(deep=True)
    result[output] = pd.to_numeric(result[output], errors="coerce").clip(
        lower=float(lower),
        upper=float(upper),
    )
    return result


def disaggregate_us_puf_aggregate_records_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext,
) -> pd.DataFrame:
    """Execute the US PUF aggregate-row disaggregation manifest operation."""

    if operation.kind != "disaggregate_aggregate_records":
        raise SourceRuntimeError(
            "PUF aggregate-row handler received unexpected operation "
            f"{operation.kind!r}."
        )
    params = operation.parameters
    unexpected = sorted(set(params) - _PUF_AGGREGATE_DISAGGREGATION_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation received unsupported "
            f"parameter(s): {unexpected}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires a current source frame."
        )
    if params.get("spec") != "puf_aggregate_record_disaggregation":
        raise SourceRuntimeError(
            "US disaggregate_aggregate_records currently supports only "
            "spec='puf_aggregate_record_disaggregation'."
        )
    if params.get("method") != "donor_template_calibration":
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires "
            "method='donor_template_calibration'."
        )

    spec = load_default_puf_aggregate_disaggregation_spec()
    replace_records = tuple(int(value) for value in params.get("replace_records", ()))
    if replace_records != spec.aggregate_recids:
        raise SourceRuntimeError(
            "PUF aggregate-record manifest replace_records do not match the "
            f"packaged spec: {replace_records} != {spec.aggregate_recids}."
        )

    weight = params.get("weight")
    if weight not in {"s006", "S006"}:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires weight='s006'."
        )
    if params.get("amount_columns") != "irs_puf_amount_columns":
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires "
            "amount_columns='irs_puf_amount_columns'."
        )
    if params.get("seed_from_build_config") is not True:
        raise SourceRuntimeError(
            "PUF aggregate-record disaggregation requires seed_from_build_config=true."
        )

    return disaggregate_puf_aggregate_records(
        frame,
        seed=int(context.config.seed),
        spec=spec,
    )


def _reject_unexpected_parameters(
    params: Mapping[str, object],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unexpected = sorted(set(params) - allowed)
    if unexpected:
        raise SourceRuntimeError(
            f"{label} received unsupported parameter(s): {unexpected}."
        )


def _required_string_param(
    params: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise SourceRuntimeError(f"{label} requires a non-empty {key!r} parameter.")
    return value


def _string_param_with_default(
    params: Mapping[str, object],
    key: str,
    *,
    default: str,
    label: str,
) -> str:
    value = params.get(key, default)
    if not isinstance(value, str) or not value:
        raise SourceRuntimeError(f"{label} requires {key!r} to be a string.")
    return value


def _optional_string_param(
    params: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SourceRuntimeError(f"{label} requires {key!r} to be a string or null.")
    return value


def _required_string_sequence_param(
    params: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> tuple[str, ...]:
    value = params.get(key)
    if (
        not isinstance(value, list | tuple)
        or isinstance(value, str | bytes | bytearray)
        or not value
    ):
        raise SourceRuntimeError(
            f"{label} requires {key!r} to be a non-empty list of strings."
        )
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise SourceRuntimeError(
                f"{label} requires {key!r} to contain only non-empty strings."
            )
        result.append(item)
    return tuple(result)


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(f"{label} is missing required column(s): {missing}.")


def _binary_calibration_value_state(
    frame: pd.DataFrame,
    variable: str,
    *,
    label: str,
) -> _BinaryCalibrationValueState:
    if variable.isidentifier():
        _require_columns(frame, [variable], label=f"{label} frame")
        return _BinaryCalibrationValueState(column=variable)

    if "<" in variable and "<=" not in variable:
        column, raw_threshold = [piece.strip() for piece in variable.split("<", 1)]
        if not column:
            raise SourceRuntimeError(f"{label} has an invalid comparison {variable!r}.")
        _require_columns(frame, [column], label=f"{label} frame")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise SourceRuntimeError(
                f"{label} comparison has a nonnumeric threshold: {variable!r}."
            ) from exc
        return _BinaryCalibrationValueState(
            column=column,
            less_than_threshold=threshold,
        )

    raise SourceRuntimeError(
        f"{label} requires a stored boolean column or simple "
        f"'column < number' expression, got {variable!r}."
    )


def _preserved_true_mask(
    frame: pd.DataFrame,
    params: Mapping[str, object],
    *,
    label: str,
) -> pd.Series:
    if not bool(params.get("preserve_true_anchors", False)):
        return pd.Series(False, index=frame.index)
    anchor = params.get("preserve_true_anchor")
    if not isinstance(anchor, str) or not anchor:
        raise SourceRuntimeError(
            f"{label} preserve_true_anchors requires a preserve_true_anchor column."
        )
    _require_columns(frame, [anchor], label=f"{label} frame")
    return frame[anchor].fillna(False).astype(bool)


def _stable_draws(frame: pd.DataFrame, *, seed: int, salt: str) -> pd.Series:
    """Seeded uniform draws keyed by ``tax_unit_id`` when present.

    Without a ``tax_unit_id`` column the draws fall back to the positional row
    index, which is not identity-stable across frame reorderings. Stages at
    other grains (e.g. person) should mint their own identity-keyed draw
    column and name it via the operation's ``draw`` parameter rather than
    relying on this fallback.
    """

    if "tax_unit_id" in frame:
        keys = frame["tax_unit_id"]
    else:
        keys = pd.Series(frame.index, index=frame.index)
    denominator = float(2**64)
    draws = [
        int.from_bytes(
            hashlib.blake2b(
                f"{seed}:{salt}:{key}".encode(),
                digest_size=8,
            ).digest(),
            byteorder="big",
            signed=False,
        )
        / denominator
        for key in keys
    ]
    return pd.Series(draws, index=frame.index, dtype=float)


def _first_available_target_table(
    target_names: tuple[str, ...],
    context: SourceRuntimeContext,
    *,
    label: str,
) -> tuple[str, pd.DataFrame]:
    for target_name in target_names:
        if target_name in context.tables:
            return target_name, context.read_table(target_name)
    raise SourceRuntimeError(
        f"{label} requires one provided target table from "
        f"{list(target_names)}; available tables: {sorted(context.tables)}."
    )


def _target_value_column(targets: pd.DataFrame, *, label: str) -> str:
    for column in _TARGET_VALUE_COLUMNS:
        if column in targets.columns:
            return column
    raise SourceRuntimeError(
        f"{label} must include one target-value column from "
        f"{list(_TARGET_VALUE_COLUMNS)}."
    )


def _group_columns(
    declared: object,
    targets: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    label: str,
) -> tuple[str, ...]:
    if declared is not None:
        if isinstance(declared, str):
            return (declared,)
        if isinstance(declared, list | tuple) and not isinstance(
            declared,
            str | bytes | bytearray,
        ):
            return tuple(str(item) for item in declared)
        raise SourceRuntimeError(
            f"{label} group_by must be a string or list of strings."
        )
    return tuple(
        column
        for column in ("state_fips",)
        if column in targets.columns and column in frame.columns
    )


def _domain_mask(frame: pd.DataFrame, domain: object, *, label: str) -> pd.Series:
    if domain is None:
        return pd.Series(True, index=frame.index)
    if isinstance(domain, str) and domain in frame.columns:
        return frame[domain].fillna(False).astype(bool)
    if isinstance(domain, str) and ">" in domain:
        column, raw_threshold = [piece.strip() for piece in domain.split(">", 1)]
        if column in frame.columns:
            threshold = float(raw_threshold)
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > threshold
    raise SourceRuntimeError(
        f"{label} domain must be a boolean column or a simple "
        f"'column > number' expression, got {domain!r}."
    )
