"""CPS ASEC workers' compensation carried from measured annual ``WC_VAL``.

The retired eCPS pipeline mapped ``WC_VAL`` directly at person grain and kept
workers' compensation out of its separate two-slot disability-benefits sum.
It then used the common eight-predictor, at-most-5,000-person QRF to replace
this CPS-only leaf on the PUF support half.  This port preserves that source
separation and uses typed person weights under Populace's weighting contract.
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
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "WORKERS_COMPENSATION_ARCHIVED_DERIVATION_URL",
    "WORKERS_COMPENSATION_ARCHIVED_PUF_IMPUTATION_URL",
    "WORKERS_COMPENSATION_ARCHIVED_PUF_OUTPUTS_URL",
    "WORKERS_COMPENSATION_ARCHIVED_SOURCE_COLUMNS_URL",
    "US_WORKERS_COMPENSATION_NONCONSTANT_PERSON_COLUMNS",
    "US_WORKERS_COMPENSATION_OUTPUT_COLUMNS",
    "US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS",
    "US_WORKERS_COMPENSATION_STAGE_NAME",
    "derive_us_workers_compensation_from_asec",
    "derive_us_workers_compensation_from_manifest",
    "impute_us_workers_compensation_to_puf_support_from_manifest",
    "us_workers_compensation_signal_gate",
    "us_workers_compensation_stage_spec",
    "us_workers_compensation_summary",
    "with_us_workers_compensation",
]

QRF: Any | None = None

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
WORKERS_COMPENSATION_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1559-L1571"
)
WORKERS_COMPENSATION_ARCHIVED_SOURCE_COLUMNS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/census_cps.py#L306-L381"
)
WORKERS_COMPENSATION_ARCHIVED_PUF_OUTPUTS_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L135-L194"
)
WORKERS_COMPENSATION_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L639-L745"
)

US_WORKERS_COMPENSATION_STAGE_NAME = "workers_compensation_input"
US_WORKERS_COMPENSATION_OUTPUT_COLUMNS: tuple[str, ...] = ("workers_compensation",)
US_WORKERS_COMPENSATION_NONCONSTANT_PERSON_COLUMNS = (
    US_WORKERS_COMPENSATION_OUTPUT_COLUMNS
)
US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = ("WC_VAL",)

_OUTPUT = US_WORKERS_COMPENSATION_OUTPUT_COLUMNS[0]
_PERSON_WEIGHT_COLUMN = "person_weight"
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
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
_PREDICTOR_PREFIX = "workers_compensation_predictor_"
_EXPECTED_DIRECT_PARAMETERS = {
    "source": "WC_VAL",
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
_NONZERO_SHARE_BAND = (0.0005, 0.05)
_CHANNEL_NONZERO_SHARE_BANDS = {
    _BASE_ASEC_SUPPORT_CHANNEL: (0.001, 0.01),
    _PUF_TAX_DETAIL_SUPPORT_CHANNEL: (0.0002, 0.10),
}


def us_workers_compensation_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged workers-compensation stage."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_WORKERS_COMPENSATION_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_WORKERS_COMPENSATION_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_WORKERS_COMPENSATION_STAGE_NAME]
    missing = sorted(set(US_WORKERS_COMPENSATION_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_WORKERS_COMPENSATION_STAGE_NAME!r} manifest stage does not "
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
            f"US workers-compensation derivation requires ASEC source column {column!r}."
        )
    values = pd.to_numeric(person[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US workers-compensation source {column!r} contains {nonfinite} "
            "nonnumeric or nonfinite value(s)."
        )
    if nonnegative:
        negative = int(np.count_nonzero(values < 0.0))
        if negative:
            raise SourceRuntimeError(
                f"US workers-compensation source {column!r} contains {negative} "
                "negative value(s)."
            )
    return values


def derive_us_workers_compensation_from_asec(
    person: pd.DataFrame,
    *,
    source: str = "WC_VAL",
    output_column: str = _OUTPUT,
) -> pd.DataFrame:
    """Apply the retired direct annual ``WC_VAL`` mapping."""

    result = person.copy(deep=True)
    result[output_column] = _strict_numeric_source(
        person,
        source,
        nonnegative=True,
    )
    return result


def derive_us_workers_compensation_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Interpret the manifest's exact ASEC ``WC_VAL`` derivation."""

    if operation.kind != "derive_workers_compensation":
        raise SourceRuntimeError(
            "US workers-compensation derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US workers-compensation derivation requires the person table first."
        )
    parameters = dict(operation.parameters)
    if parameters != _EXPECTED_DIRECT_PARAMETERS:
        raise SourceRuntimeError(
            "US workers-compensation derivation drifted from the archived method: "
            f"expected {_EXPECTED_DIRECT_PARAMETERS}, got {parameters}."
        )
    return derive_us_workers_compensation_from_asec(
        frame,
        source=parameters["source"],
        output_column=parameters["output"],
    )


def impute_us_workers_compensation_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """QRF-impute the CPS-only leaf onto PUF-clone people."""

    if operation.kind != "impute_workers_compensation_to_puf_support":
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation requires the person table first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation parameters must match the "
            f"archived method; missing={missing_parameters}, "
            f"unexpected={unexpected}."
        )
    if _PERSON_SUPPORT_CHANNEL_COLUMN not in frame.columns:
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PREDICTORS:
        raise SourceRuntimeError(
            "US workers-compensation PUF predictors drifted from the archived "
            f"method: expected {list(_PREDICTORS)}, got {list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation must use typed person weights."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation seed must come from the build "
            "config."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError(
            "US workers-compensation PUF max_train_samples and n_estimators must "
            "be positive."
        )

    predictor_columns = [_PREDICTOR_PREFIX + name for name in predictors]
    required = [_PERSON_WEIGHT_COLUMN, *predictor_columns, _OUTPUT]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"US workers-compensation PUF imputation is missing column(s): {missing}."
        )

    channel = frame[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str)
    asec_mask = channel == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = channel == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation requires nonempty ASEC and "
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
            "US workers-compensation QRF person weights must be finite and nonnegative."
        )
    if float(numeric_weights.sum()) <= 0.0:
        raise SourceRuntimeError(
            "US workers-compensation QRF person weights sum to zero."
        )

    if len(training) > max_train_samples:
        sample = training.sample(
            n=max_train_samples,
            random_state=(context.config.seed if context is not None else 0),
        ).index
        training = training.loc[sample]
        weights = weights.loc[sample]

    for column in (*predictors, _OUTPUT):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US workers-compensation QRF training column {column!r} "
                "contains nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US workers-compensation QRF prediction column {column!r} "
                "contains nonfinite values."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("populace.fit").QRF
    seed = context.config.seed if context is not None else 0
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        training,
        list(predictors),
        [_OUTPUT],
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test)
    if _OUTPUT not in predictions:
        raise SourceRuntimeError(
            f"US workers-compensation QRF prediction is missing {_OUTPUT!r}."
        )
    predicted = pd.to_numeric(predictions[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(predicted).all():
        raise SourceRuntimeError(
            f"US workers-compensation QRF produced nonfinite {_OUTPUT} values."
        )
    if bool((predicted < 0.0).any()):
        raise SourceRuntimeError(
            f"US workers-compensation QRF produced negative {_OUTPUT} values."
        )
    result = frame.copy(deep=True)
    result.loc[puf_mask, _OUTPUT] = predicted
    return result


def _person_workers_compensation_predictors(frame: Frame) -> pd.DataFrame:
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
                        "US workers-compensation QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation cannot construct a predictor "
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
            "US workers-compensation PUF imputation requires is_male, is_female, "
            "or measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError(
            "US workers-compensation PUF imputation requires has_esi."
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
            "US workers-compensation PUF imputation requires tax-unit filing status."
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
            "US workers-compensation PUF imputation requires tax_unit_role_input "
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


def with_us_workers_compensation(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    allow_existing_without_source: bool = False,
) -> Frame:
    """Materialize direct ASEC and post-PUF-clone workers' compensation."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US workers' compensation require the US schema.")
    person = frame.table("person")
    source_available = all(
        column in person for column in US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS
    )
    if not source_available:
        if allow_existing_without_source and _surface_carries_signal(frame):
            return frame
        missing = [
            column
            for column in US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS
            if column not in person
        ]
        raise ValueError(
            "US workers-compensation stage cannot heal a default surface without "
            f"measured ASEC source column(s): {missing}."
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if _PERSON_SUPPORT_CHANNEL_COLUMN in person:
        predictors = _person_workers_compensation_predictors(frame)
        for column in _PREDICTORS:
            stage_person[_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_workers_compensation_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_workers_compensation": (
                derive_us_workers_compensation_from_manifest
            ),
            "impute_workers_compensation_to_puf_support": (
                impute_us_workers_compensation_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[_OUTPUT].isna().any():
        raise ValueError(
            f"US workers-compensation stage output {_OUTPUT!r} does not cover "
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
    )


def us_workers_compensation_summary(frame: Frame) -> dict[str, object]:
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
    if _PERSON_SUPPORT_CHANNEL_COLUMN in person:
        channel = person[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str).to_numpy()
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
    if "WC_VAL" in person:
        source = pd.to_numeric(person["WC_VAL"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        source_mask = np.ones(len(person), dtype=bool)
        if _PERSON_SUPPORT_CHANNEL_COLUMN in person:
            source_mask = (
                person[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str).to_numpy()
                == _BASE_ASEC_SUPPORT_CHANNEL
            )
        source_valid = np.isfinite(source) & (source >= 0.0)
        summary["source_invalid"] = int(np.count_nonzero(source_mask & ~source_valid))
        summary["source_mismatch_count"] = int(
            np.count_nonzero(
                source_mask
                & source_valid
                & finite
                & ~np.isclose(values, source, rtol=0.0, atol=0.0)
            )
        )
    return summary


def us_workers_compensation_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, nondefault signal on both support halves."""

    person = frame.table("person")
    if _OUTPUT not in person:
        return GateResult(
            name="workers_compensation_signal",
            passed=False,
            failures=(f"person column missing: {_OUTPUT}.",),
            details={"missing": [_OUTPUT]},
        )

    summary = us_workers_compensation_summary(frame)
    failures: list[str] = []
    if summary["nonfinite"]:
        failures.append(f"{_OUTPUT}: {int(summary['nonfinite'])} nonfinite values.")
    if summary["negative"]:
        failures.append(f"{_OUTPUT}: {int(summary['negative'])} negative values.")
    if int(summary.get("source_invalid", 0)):
        failures.append(
            f"WC_VAL: {int(summary['source_invalid'])} invalid ASEC source values."
        )
    if int(summary.get("source_mismatch_count", 0)):
        failures.append(
            f"{_OUTPUT}: {int(summary['source_mismatch_count'])} ASEC WC_VAL "
            "reconciliation mismatch(es)."
        )
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
        name="workers_compensation_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _surface_carries_signal(frame: Frame) -> bool:
    return (
        _OUTPUT in frame.table("person")
        and us_workers_compensation_signal_gate(frame).passed
    )
