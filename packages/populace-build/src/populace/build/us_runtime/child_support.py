"""CPS ASEC child-support inputs with retired PUF-clone QRF treatment.

The retired eCPS pipeline carried annual person-level child support received
directly from ASEC ``CSP_VAL`` and annual child support paid from ``CHSP_VAL``.
After its PUF support clone received PUF-imputed income, one second-stage QRF
jointly replaced both child-support leaves on the clone half using eight
documented predictors and at most 5,000 ASEC training people.  Immutable
archived source coordinates are exposed below.

This port preserves the annual positive-dollar semantics, observed top-coded
values, and person grain.  It does not balance paid against received, negate
expenses, impose mutual exclusivity, or invent a child/household eligibility
mask.  The QRF fit uses typed person weights, deliberately strengthening the
archived unweighted fit under Populace's build-wide weighting contract.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL",
    "CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL",
    "CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL",
    "CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL",
    "US_CHILD_SUPPORT_NONCONSTANT_PERSON_COLUMNS",
    "US_CHILD_SUPPORT_OUTPUT_COLUMNS",
    "US_CHILD_SUPPORT_REQUIRED_SOURCE_COLUMNS",
    "US_CHILD_SUPPORT_STAGE_NAME",
    "derive_us_child_support_from_asec",
    "derive_us_child_support_from_manifest",
    "impute_us_child_support_to_puf_support_from_manifest",
    "us_child_support_signal_gate",
    "us_child_support_stage_spec",
    "us_child_support_summary",
    "with_us_child_support_inputs",
]

QRF: Any | None = None

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1493-L1496"
)
CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1572-L1574"
)
CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L135-L194"
)
CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L639-L745"
)

US_CHILD_SUPPORT_STAGE_NAME = "child_support_inputs"
US_CHILD_SUPPORT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "child_support_received",
    "child_support_expense",
)
US_CHILD_SUPPORT_NONCONSTANT_PERSON_COLUMNS = US_CHILD_SUPPORT_OUTPUT_COLUMNS
US_CHILD_SUPPORT_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "CSP_VAL",
    "CHSP_VAL",
)

_PERSON_WEIGHT_COLUMN = "person_weight"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)
_PREDICTOR_PREFIX = "child_support_predictor_"
_EXPECTED_DIRECT_PARAMETERS = {
    "received_source": "CSP_VAL",
    "received_output": "child_support_received",
    "expense_source": "CHSP_VAL",
    "expense_output": "child_support_expense",
}
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {
        "predictors",
        "max_train_samples",
        "n_estimators",
        "seed_from_build_config",
        "weight",
    }
)
#: Per-column plausibility bands on the pooled (ASEC + PUF-clone) person share.
#: The halves are NOT symmetric and must not share a floor: CPS asks custodial
#: parents about receipt but relies on payer self-reports for expense, so the
#: expense share runs structurally lower. Measured at the first full-scale run
#: through this gate (Build M base, 2026-07-13, seed 0): raw ASEC person share
#: 0.66% expense / 0.98% received; the QRF draws a covariate-consistent 0.27%
#: on the PUF tax-detail clone channel; the ~50/50 pool blends to 0.466%
#: expense — non-degenerate, deterministic, and reproduced bit-identically with
#: the leaf-storage guard disabled. A shared 0.005 floor mislabeled that as
#:  degenerate. The retired us-data pipeline's sequential imputation shipped
#: 1.99% (extended CPS) and 2.96% (enhanced CPS) against the same 0.62-0.66%
#: survey marginal — prevalence drift, not a reference; whether prevalence
#: should instead be seeded to OCSE payer counts is populace#417. Degeneracy
#: per channel is still enforced by _CHANNEL_NONZERO_SHARE_BAND below.
_NONZERO_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "child_support_received": (0.005, 0.15),
    "child_support_expense": (0.003, 0.15),
}
_CHANNEL_NONZERO_SHARE_BAND = (0.002, 0.15)


def us_child_support_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged child-support stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_CHILD_SUPPORT_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_CHILD_SUPPORT_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_CHILD_SUPPORT_STAGE_NAME]
    missing = sorted(set(US_CHILD_SUPPORT_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_CHILD_SUPPORT_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _strict_nonnegative_source(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        raise SourceRuntimeError(
            f"US child-support derivation requires ASEC source column {column!r}."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US child-support source {column!r} contains {nonfinite} nonnumeric "
            "or nonfinite value(s)."
        )
    negative = int(np.count_nonzero(values < 0.0))
    if negative:
        raise SourceRuntimeError(
            f"US child-support source {column!r} contains {negative} negative value(s)."
        )
    return values


def derive_us_child_support_from_asec(
    person: pd.DataFrame,
    *,
    received_source_column: str = "CSP_VAL",
    received_output_column: str = "child_support_received",
    expense_source_column: str = "CHSP_VAL",
    expense_output_column: str = "child_support_expense",
) -> pd.DataFrame:
    """Carry both measured annual ASEC child-support fields exactly."""

    received = _strict_nonnegative_source(person, received_source_column)
    expense = _strict_nonnegative_source(person, expense_source_column)
    result = person.copy(deep=True)
    result[received_output_column] = received
    result[expense_output_column] = expense
    return result


def derive_us_child_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Interpret the manifest's exact ASEC direct-carry operation."""

    if operation.kind != "derive_child_support_inputs":
        raise SourceRuntimeError(
            "US child-support derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US child-support derivation requires the person table first."
        )
    parameters = dict(operation.parameters)
    if parameters != _EXPECTED_DIRECT_PARAMETERS:
        raise SourceRuntimeError(
            "US child-support direct mapping drifted from the archived method: "
            f"expected {_EXPECTED_DIRECT_PARAMETERS}, got {parameters}."
        )
    return derive_us_child_support_from_asec(
        frame,
        received_source_column=parameters["received_source"],
        received_output_column=parameters["received_output"],
        expense_source_column=parameters["expense_source"],
        expense_output_column=parameters["expense_output"],
    )


def impute_us_child_support_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Jointly QRF-impute the two leaves onto PUF-clone people."""

    if operation.kind != "impute_child_support_to_puf_support":
        raise SourceRuntimeError(
            "US child-support PUF imputation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US child-support PUF imputation requires the person table first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US child-support PUF imputation parameters must match the archived "
            f"method; missing={missing_parameters}, unexpected={unexpected}."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PREDICTORS:
        raise SourceRuntimeError(
            "US child-support PUF predictors drifted from the archived method: "
            f"expected {list(_PREDICTORS)}, got {list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US child-support PUF imputation must use typed person weights."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US child-support PUF imputation seed must come from the build config."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError(
            "US child-support PUF max_train_samples and n_estimators must be positive."
        )

    predictor_columns = [_PREDICTOR_PREFIX + name for name in predictors]
    required = [
        _PERSON_WEIGHT_COLUMN,
        *predictor_columns,
        *US_CHILD_SUPPORT_OUTPUT_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"US child-support PUF imputation is missing column(s): {missing}."
        )

    role = support_role_series(frame, entity="person")
    asec_mask = role == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = role == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US child-support PUF imputation requires nonempty ASEC and "
            "PUF-tax-detail support channels."
        )

    training = frame.loc[
        asec_mask,
        [*predictor_columns, *US_CHILD_SUPPORT_OUTPUT_COLUMNS],
    ].copy()
    training.columns = [*predictors, *US_CHILD_SUPPORT_OUTPUT_COLUMNS]
    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = list(predictors)
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    )
    numeric_weights = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_weights).all() or bool((numeric_weights < 0.0).any()):
        raise SourceRuntimeError(
            "US child-support QRF person weights must be finite and nonnegative."
        )
    if float(numeric_weights.sum()) <= 0.0:
        raise SourceRuntimeError("US child-support QRF person weights sum to zero.")

    if len(training) > max_train_samples:
        sample = training.sample(
            n=max_train_samples,
            random_state=(context.config.seed if context is not None else 0),
        ).index
        training = training.loc[sample]
        weights = weights.loc[sample]

    for column in (*predictors, *US_CHILD_SUPPORT_OUTPUT_COLUMNS):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US child-support QRF training column {column!r} contains "
                "nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US child-support QRF prediction column {column!r} contains "
                "nonfinite values."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF
    seed = context.config.seed if context is not None else 0
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        training,
        list(predictors),
        list(US_CHILD_SUPPORT_OUTPUT_COLUMNS),
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test)
    missing_outputs = [
        output
        for output in US_CHILD_SUPPORT_OUTPUT_COLUMNS
        if output not in predictions
    ]
    if missing_outputs:
        raise SourceRuntimeError(
            f"US child-support QRF prediction is missing output(s): {missing_outputs}."
        )

    result = frame.copy(deep=True)
    for output in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        predicted = pd.to_numeric(predictions[output], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(predicted).all():
            raise SourceRuntimeError(
                f"US child-support QRF produced nonfinite {output} values."
            )
        if bool((predicted < 0.0).any()):
            raise SourceRuntimeError(
                f"US child-support QRF produced negative {output} values."
            )
        result.loc[puf_mask, output] = predicted
    return result


def _person_child_support_predictors(frame: Frame) -> pd.DataFrame:
    """Build the retired eight second-stage QRF predictors on people."""

    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    predictors = pd.DataFrame(index=person.index)

    def _numeric(*columns: str) -> np.ndarray:
        for column in columns:
            if column in person:
                values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                if not np.isfinite(values).all():
                    raise SourceRuntimeError(
                        "US child-support QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US child-support PUF imputation cannot construct a predictor from "
            f"any of {list(columns)}."
        )

    predictors["age"] = _numeric("age", "A_AGE")
    if "is_male" in person:
        predictors["is_male"] = person["is_male"].fillna(False).astype(bool)
    elif "is_female" in person:
        predictors["is_male"] = ~person["is_female"].fillna(False).astype(bool)
    elif "A_SEX" in person:
        predictors["is_male"] = _numeric("A_SEX") == 1
    else:
        raise SourceRuntimeError(
            "US child-support PUF imputation requires is_male, is_female, or "
            "measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError("US child-support PUF imputation requires has_esi.")
    predictors["has_esi"] = person["has_esi"].fillna(False).astype(bool)

    filing_status_column = next(
        (
            column
            for column in ("filing_status_input", "filing_status")
            if column in tax_unit
        ),
        None,
    )
    if filing_status_column is None:
        raise SourceRuntimeError(
            "US child-support PUF imputation requires tax-unit filing status."
        )
    filing_status = tax_unit[filing_status_column].map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )
    joint_by_id = pd.Series(
        filing_status.eq("JOINT").to_numpy(),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )
    predictors["tax_unit_is_joint"] = (
        person["person_tax_unit_id"].map(joint_by_id).fillna(False).astype(bool)
    )

    if "tax_unit_role_input" not in person:
        raise SourceRuntimeError(
            "US child-support PUF imputation requires tax_unit_role_input to "
            "count dependents."
        )
    dependent = (
        person["tax_unit_role_input"]
        .map(
            lambda value: (
                value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            )
        )
        .eq("DEPENDENT")
    )
    predictors["tax_unit_count_dependents"] = (
        dependent.groupby(person["person_tax_unit_id"])
        .transform("sum")
        .to_numpy(dtype=np.float64)
    )
    predictors["employment_income"] = _numeric(
        "employment_income_before_lsr", "WSAL_VAL"
    )
    predictors["self_employment_income"] = _numeric(
        "self_employment_income_before_lsr", "SEMP_VAL"
    )
    social_security_columns = (
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
    )
    if all(column in person for column in social_security_columns):
        predictors["social_security"] = np.sum(
            np.column_stack([_numeric(column) for column in social_security_columns]),
            axis=1,
        )
    else:
        predictors["social_security"] = _numeric("SS_VAL")
    return predictors


def with_us_child_support_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    allow_existing_without_source: bool = False,
) -> Frame:
    """Materialize direct ASEC and post-PUF-clone child-support inputs."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US child-support inputs require the US schema.")
    person = frame.table("person")
    source_available = all(
        column in person for column in US_CHILD_SUPPORT_REQUIRED_SOURCE_COLUMNS
    )
    if not source_available:
        if allow_existing_without_source and _child_support_surface_carries_signal(
            frame
        ):
            return frame
        missing = [
            column
            for column in US_CHILD_SUPPORT_REQUIRED_SOURCE_COLUMNS
            if column not in person
        ]
        raise ValueError(
            "US child-support stage cannot heal a default surface without "
            f"measured ASEC source column(s): {missing}."
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_role_metadata(person, entity="person"):
        predictors = _person_child_support_predictors(frame)
        for column in _PREDICTORS:
            stage_person[_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_child_support_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_child_support_inputs": derive_us_child_support_from_manifest,
            "impute_child_support_to_puf_support": (
                impute_us_child_support_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                f"US child-support stage output {column!r} does not cover every person."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_child_support_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics for both leaves."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    channel = (
        support_role_series(person, entity="person").to_numpy()
        if has_support_role_metadata(person, entity="person")
        else None
    )
    columns: dict[str, dict[str, object]] = {}
    for output in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[output], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(values)
        positive = finite & (values > 0.0)
        detail: dict[str, object] = {
            "positive_share": (
                float(weights[positive].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
            "positive_share_band": list(_NONZERO_SHARE_BANDS[output]),
            "weighted_total": float((np.nan_to_num(values) * weights).sum()),
            "nonfinite": int(np.count_nonzero(~finite)),
            "negative": int(np.count_nonzero(finite & (values < 0.0))),
        }
        if channel is not None:
            channels: dict[str, dict[str, float | int]] = {}
            for name in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL):
                mask = channel == name
                channel_weight = float(weights[mask].sum())
                channels[name] = {
                    "positive_rows": int(np.count_nonzero(mask & positive)),
                    "positive_share": (
                        float(weights[mask & positive].sum()) / channel_weight
                        if channel_weight > 0.0
                        else 0.0
                    ),
                    "weighted_total": float(
                        (np.nan_to_num(values[mask]) * weights[mask]).sum()
                    ),
                }
            detail["channels"] = channels
        columns[output] = detail
    return {"columns": columns}


def us_child_support_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, nondefault signal on both support halves."""

    person = frame.table("person")
    missing = [
        output for output in US_CHILD_SUPPORT_OUTPUT_COLUMNS if output not in person
    ]
    if missing:
        return GateResult(
            name="child_support_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_child_support_summary(frame)
    failures: list[str] = []
    columns = summary["columns"]
    for output in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        detail = columns[output]
        if detail["nonfinite"]:
            failures.append(f"{output}: {int(detail['nonfinite'])} nonfinite values.")
        if detail["negative"]:
            failures.append(f"{output}: {int(detail['negative'])} negative values.")
        share = float(detail["positive_share"])
        low, high = detail["positive_share_band"]
        if not (low <= share <= high):
            failures.append(
                f"{output}: nonzero share {share:.4f} outside plausibility "
                f"band [{low}, {high}]."
            )
        channels = detail.get("channels")
        if isinstance(channels, dict):
            channel_low, channel_high = _CHANNEL_NONZERO_SHARE_BAND
            for name in (
                _BASE_ASEC_SUPPORT_CHANNEL,
                _PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ):
                channel_detail = channels.get(name)
                if not isinstance(channel_detail, dict):
                    failures.append(f"{output}: missing {name} channel diagnostics.")
                    continue
                channel_share = float(channel_detail["positive_share"])
                if not (channel_low <= channel_share <= channel_high):
                    failures.append(
                        f"{output}: {name} nonzero share {channel_share:.4f} "
                        f"outside plausibility band [{channel_low}, {channel_high}]."
                    )
    return GateResult(
        name="child_support_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _child_support_surface_carries_signal(frame: Frame) -> bool:
    if any(
        output not in frame.table("person")
        for output in US_CHILD_SUPPORT_OUTPUT_COLUMNS
    ):
        return False
    return us_child_support_signal_gate(frame).passed
