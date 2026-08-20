"""CPS ASEC non-SSA, non-workers-comp disability benefit input.

The retired eCPS pipeline summed the two annual ASEC disability-income slots
only when their source code was not 1 (workers' compensation).  It then used
the common eight-predictor, at-most-5,000-person QRF to replace this CPS-only
leaf on the PUF support half.  Immutable archived coordinates are exposed
below.

This port preserves positive annual dollars at person grain.  It does not fold
Social Security disability or workers' compensation into this leaf, and it
does not invent eligibility masks or rebalance the two source slots.  The QRF
uses typed person weights under Microcosm's build-wide weighting contract.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    load_source_manifest,
)
from microcosm.build.source_runtime import (
    SourceRNGCapability,
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.build.us_runtime.support_provenance import (
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL",
    "DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL",
    "DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL",
    "DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL",
    "US_DISABILITY_BENEFITS_NONCONSTANT_PERSON_COLUMNS",
    "US_DISABILITY_BENEFITS_OUTPUT_COLUMNS",
    "US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS",
    "US_DISABILITY_BENEFITS_STAGE_NAME",
    "derive_us_disability_benefits_from_asec",
    "derive_us_disability_benefits_from_manifest",
    "impute_us_disability_benefits_to_puf_support_from_manifest",
    "us_disability_benefits_signal_gate",
    "us_disability_benefits_stage_spec",
    "us_disability_benefits_summary",
    "with_us_disability_benefits",
]

QRF: Any | None = None

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1561-L1571"
)
DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/census_cps.py#L306-L381"
)
DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L135-L194"
)
DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L639-L745"
)

US_DISABILITY_BENEFITS_STAGE_NAME = "disability_benefits_input"
US_DISABILITY_BENEFITS_OUTPUT_COLUMNS: tuple[str, ...] = ("disability_benefits",)
US_DISABILITY_BENEFITS_NONCONSTANT_PERSON_COLUMNS = (
    US_DISABILITY_BENEFITS_OUTPUT_COLUMNS
)
US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "DIS_VAL1",
    "DIS_SC1",
    "DIS_VAL2",
    "DIS_SC2",
)

_OUTPUT = US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]
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
_PREDICTOR_PREFIX = "disability_benefits_predictor_"
_EXPECTED_DIRECT_PARAMETERS = {
    "first_amount_source": "DIS_VAL1",
    "first_code_source": "DIS_SC1",
    "second_amount_source": "DIS_VAL2",
    "second_code_source": "DIS_SC2",
    "workers_compensation_code": 1,
    "output": _OUTPUT,
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
# Broad source-plausibility bounds: reject default/near-universal surfaces,
# while leaving the exact raw-source and support-channel tests to pin semantics.
_NONZERO_SHARE_BAND = (0.003, 0.15)
_CHANNEL_NONZERO_SHARE_BANDS = {
    _BASE_ASEC_SUPPORT_CHANNEL: (0.003, 0.02),
    _PUF_TAX_DETAIL_SUPPORT_CHANNEL: (0.002, 0.15),
}


def us_disability_benefits_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged disability-benefits stage."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_DISABILITY_BENEFITS_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_DISABILITY_BENEFITS_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_DISABILITY_BENEFITS_STAGE_NAME]
    missing = sorted(set(US_DISABILITY_BENEFITS_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_DISABILITY_BENEFITS_STAGE_NAME!r} manifest stage does not "
            f"declare output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _strict_numeric_source(
    person: pd.DataFrame,
    column: str,
    *,
    nonnegative: bool,
) -> np.ndarray:
    if column not in person.columns:
        raise SourceRuntimeError(
            f"US disability-benefits derivation requires ASEC source column {column!r}."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US disability-benefits source {column!r} contains {nonfinite} "
            "nonnumeric or nonfinite value(s)."
        )
    if nonnegative:
        negative = int(np.count_nonzero(values < 0.0))
        if negative:
            raise SourceRuntimeError(
                f"US disability-benefits source {column!r} contains {negative} "
                "negative value(s)."
            )
    return values


def derive_us_disability_benefits_from_asec(
    person: pd.DataFrame,
    *,
    first_amount_source: str = "DIS_VAL1",
    first_code_source: str = "DIS_SC1",
    second_amount_source: str = "DIS_VAL2",
    second_code_source: str = "DIS_SC2",
    workers_compensation_code: int = 1,
    output_column: str = _OUTPUT,
) -> pd.DataFrame:
    """Apply the retired two-slot, non-workers-compensation annual sum."""

    first_amount = _strict_numeric_source(
        person,
        first_amount_source,
        nonnegative=True,
    )
    first_code = _strict_numeric_source(
        person,
        first_code_source,
        nonnegative=False,
    )
    second_amount = _strict_numeric_source(
        person,
        second_amount_source,
        nonnegative=True,
    )
    second_code = _strict_numeric_source(
        person,
        second_code_source,
        nonnegative=False,
    )
    result = person.copy(deep=True)
    result[output_column] = first_amount * (
        first_code != workers_compensation_code
    ) + second_amount * (second_code != workers_compensation_code)
    return result


def derive_us_disability_benefits_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Interpret the manifest's exact ASEC two-slot derivation."""

    if operation.kind != "derive_disability_benefits":
        raise SourceRuntimeError(
            "US disability-benefits derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US disability-benefits derivation requires the person table first."
        )
    parameters = dict(operation.parameters)
    if parameters != _EXPECTED_DIRECT_PARAMETERS:
        raise SourceRuntimeError(
            "US disability-benefits derivation drifted from the archived method: "
            f"expected {_EXPECTED_DIRECT_PARAMETERS}, got {parameters}."
        )
    return derive_us_disability_benefits_from_asec(
        frame,
        first_amount_source=parameters["first_amount_source"],
        first_code_source=parameters["first_code_source"],
        second_amount_source=parameters["second_amount_source"],
        second_code_source=parameters["second_code_source"],
        workers_compensation_code=int(parameters["workers_compensation_code"]),
        output_column=parameters["output"],
    )


def impute_us_disability_benefits_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """QRF-impute the CPS-only leaf onto PUF-clone people."""

    if operation.kind != "impute_disability_benefits_to_puf_support":
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation requires the person table first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation parameters must match the "
            f"archived method; missing={missing_parameters}, "
            f"unexpected={unexpected}."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PREDICTORS:
        raise SourceRuntimeError(
            "US disability-benefits PUF predictors drifted from the archived "
            f"method: expected {list(_PREDICTORS)}, got {list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation must use typed person weights."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation seed must come from the build "
            "config."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError(
            "US disability-benefits PUF max_train_samples and n_estimators must "
            "be positive."
        )

    predictor_columns = [_PREDICTOR_PREFIX + name for name in predictors]
    required = [_PERSON_WEIGHT_COLUMN, *predictor_columns, _OUTPUT]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"US disability-benefits PUF imputation is missing column(s): {missing}."
        )

    role = support_role_series(frame, entity="person")
    asec_mask = role == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = role == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation requires nonempty ASEC and "
            "PUF-tax-detail support channels."
        )

    training = frame.loc[asec_mask, [*predictor_columns, _OUTPUT]].copy()
    training.columns = [*predictors, _OUTPUT]
    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = list(predictors)
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    )
    numeric_weights = weights.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_weights).all() or bool((numeric_weights < 0.0).any()):
        raise SourceRuntimeError(
            "US disability-benefits QRF person weights must be finite and nonnegative."
        )
    if float(numeric_weights.sum()) <= 0.0:
        raise SourceRuntimeError(
            "US disability-benefits QRF person weights sum to zero."
        )

    if len(training) > max_train_samples:
        if context is None or context.rng is None:
            sample = training.sample(
                n=max_train_samples,
                random_state=(context.config.seed if context is not None else 0),
            ).index
        else:
            sample = context.rng.pandas_sample(
                context.rng.token("disability_benefits_training_cap"),
                training,
                n=max_train_samples,
            ).index
        training = training.loc[sample]
        weights = weights.loc[sample]

    for column in (*predictors, _OUTPUT):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US disability-benefits QRF training column {column!r} "
                "contains nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US disability-benefits QRF prediction column {column!r} "
                "contains nonfinite values."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("microcosm.fit").QRF
    seed = context.config.seed if context is not None else 0
    qrf_generators = None
    if context is not None and context.rng is not None:
        qrf_generators = context.rng.qrf_generators(
            context.rng.token("disability_benefits_puf_qrf_model")
        )
    model = QRF(n_estimators=n_estimators, seed=seed)
    fit_kwargs = (
        {} if qrf_generators is None else {"rng_generators": qrf_generators}
    )
    fitted = model.fit(
        training,
        list(predictors),
        [_OUTPUT],
        weights=weights.to_numpy(dtype=np.float64),
        **fit_kwargs,
    )
    predictions = fitted.predict(test)
    if _OUTPUT not in predictions:
        raise SourceRuntimeError(
            f"US disability-benefits QRF prediction is missing {_OUTPUT!r}."
        )
    predicted = pd.to_numeric(predictions[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(predicted).all():
        raise SourceRuntimeError(
            f"US disability-benefits QRF produced nonfinite {_OUTPUT} values."
        )
    if bool((predicted < 0.0).any()):
        raise SourceRuntimeError(
            f"US disability-benefits QRF produced negative {_OUTPUT} values."
        )
    result = frame.copy(deep=True)
    result.loc[puf_mask, _OUTPUT] = predicted
    return result


def _person_disability_benefits_predictors(frame: Frame) -> pd.DataFrame:
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
                        "US disability-benefits QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation cannot construct a predictor "
            f"from any of {list(columns)}."
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
            "US disability-benefits PUF imputation requires is_male, is_female, "
            "or measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError(
            "US disability-benefits PUF imputation requires has_esi."
        )
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
            "US disability-benefits PUF imputation requires tax-unit filing status."
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
            "US disability-benefits PUF imputation requires tax_unit_role_input "
            "to count dependents."
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


def with_us_disability_benefits(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    allow_existing_without_source: bool = False,
    rng: SourceRNGCapability | None = None,
) -> Frame:
    """Materialize direct ASEC and post-PUF-clone disability benefits."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US disability benefits require the US schema.")
    person = frame.table("person")
    source_available = all(
        column in person for column in US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS
    )
    if not source_available:
        if allow_existing_without_source and _surface_carries_signal(frame):
            return frame
        missing = [
            column
            for column in US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS
            if column not in person
        ]
        raise ValueError(
            "US disability-benefits stage cannot heal a default surface without "
            f"measured ASEC source column(s): {missing}."
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_role_metadata(person, entity="person"):
        predictors = _person_disability_benefits_predictors(frame)
        for column in _PREDICTORS:
            stage_person[_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_disability_benefits_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_disability_benefits": (derive_us_disability_benefits_from_manifest),
            "impute_disability_benefits_to_puf_support": (
                impute_us_disability_benefits_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
        rng=rng,
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[_OUTPUT].isna().any():
        raise ValueError(
            f"US disability-benefits stage output {_OUTPUT!r} does not cover "
            "every person."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_OUTPUT] = aligned[_OUTPUT].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_disability_benefits_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    values = pd.to_numeric(person[_OUTPUT], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    summary: dict[str, object] = {
        "positive_share": (
            float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
        ),
        "positive_share_band": list(_NONZERO_SHARE_BAND),
        "weighted_total": float((np.nan_to_num(values) * weights).sum()),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }
    if has_support_role_metadata(person, entity="person"):
        channel = support_role_series(person, entity="person").to_numpy()
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
        summary["channels"] = channels
    return summary


def us_disability_benefits_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, nondefault signal on both support halves."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="disability_benefits_signal",
            passed=False,
            failures=(f"person column missing: {_OUTPUT}.",),
            details={"missing": [_OUTPUT]},
        )

    summary = us_disability_benefits_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(f"{_OUTPUT}: {int(summary['nonfinite'])} nonfinite values.")
    if summary["negative"]:
        failures.append(f"{_OUTPUT}: {int(summary['negative'])} negative values.")
    share = float(summary["positive_share"])
    low, high = summary["positive_share_band"]
    if not (low <= share <= high):
        failures.append(
            f"{_OUTPUT}: nonzero share {share:.4f} outside plausibility band "
            f"[{low}, {high}]."
        )
    channels = summary.get("channels")
    if isinstance(channels, dict):
        for name in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL):
            detail = channels.get(name)
            if not isinstance(detail, dict):
                failures.append(f"{_OUTPUT}: missing {name} channel diagnostics.")
                continue
            channel_low, channel_high = _CHANNEL_NONZERO_SHARE_BANDS[name]
            channel_share = float(detail["positive_share"])
            if not (channel_low <= channel_share <= channel_high):
                failures.append(
                    f"{_OUTPUT}: {name} nonzero share {channel_share:.4f} "
                    "outside plausibility band "
                    f"[{channel_low}, {channel_high}]."
                )
    return GateResult(
        name="disability_benefits_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _surface_carries_signal(frame: Frame) -> bool:
    return (
        _OUTPUT in frame.table("person")
        and us_disability_benefits_signal_gate(frame).passed
    )
