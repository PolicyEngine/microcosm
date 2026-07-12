"""Adjacent-year CPS ASEC earnings carried by the retired eCPS build.

The archived pipeline joins each current ASEC person to the preceding ASEC
file by ``PERIDNUM``.  It accepts a prior observation only when both Census
allocation flags are zero, treats ``-1`` and ``-9999`` as unavailable, records
whether both prior earnings amounts were observed, and otherwise falls back to
the current ASEC values.  Self-employment losses are source values, not errors,
and remain signed.

After the PUF support clone is created, the retired second-stage CPS-only QRF
jointly replaces the wage and self-employment prior-year amounts on the PUF
half using eight demographic/income predictors and a 5,000-person training
cap.  Populace preserves that treatment and strengthens it with typed person
design weights.  ``employment_income_last_year`` is formula-owned in
PolicyEngine-US 1.764.6 and the retired finalizer explicitly dropped it, so it
is retained only through the joint fit and removed from the export-ready
support frame.  The two persisted input leaves are
``self_employment_income_last_year`` and ``previous_year_income_available``.
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
    "PRIOR_YEAR_INCOME_ARCHIVED_DERIVATION_URL",
    "PRIOR_YEAR_INCOME_ARCHIVED_PUF_IMPUTATION_URL",
    "PRIOR_YEAR_INCOME_ARCHIVED_PUF_OUTPUTS_URL",
    "PRIOR_YEAR_INCOME_ARCHIVED_PUF_SPLICE_URL",
    "PRIOR_YEAR_INCOME_ARCHIVED_FORMULA_OUTPUT_URL",
    "PRIOR_YEAR_INCOME_ARCHIVED_FINALIZER_URL",
    "US_PRIOR_YEAR_INCOME_NONCONSTANT_PERSON_COLUMNS",
    "US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS",
    "US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS",
    "US_PRIOR_YEAR_INCOME_REQUIRED_SOURCE_COLUMNS",
    "US_PRIOR_YEAR_INCOME_STAGE_NAME",
    "derive_us_prior_year_income_from_manifest",
    "impute_us_prior_year_income_to_puf_support_from_manifest",
    "us_prior_year_income_signal_gate",
    "us_prior_year_income_source_reconciliation_gate",
    "us_prior_year_income_stage_spec",
    "us_prior_year_income_summary",
    "with_us_prior_year_income_inputs",
]

QRF: Any | None = None

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_PACKAGE = "policyengine_" + "us_data"
_ARCHIVED_ROOT = (
    f"https://github.com/PolicyEngine/{_ARCHIVED_REPOSITORY}/blob/"
    f"{_ARCHIVED_COMMIT}/{_ARCHIVED_PACKAGE}/datasets/cps"
)
PRIOR_YEAR_INCOME_ARCHIVED_DERIVATION_URL = _ARCHIVED_ROOT + "/cps.py#L1680-L1783"
PRIOR_YEAR_INCOME_ARCHIVED_PUF_OUTPUTS_URL = (
    _ARCHIVED_ROOT + "/extended_cps.py#L140-L194"
)
PRIOR_YEAR_INCOME_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "/extended_cps.py#L639-L745"
)
PRIOR_YEAR_INCOME_ARCHIVED_PUF_SPLICE_URL = (
    _ARCHIVED_ROOT + "/extended_cps.py#L1014-L1073"
)
PRIOR_YEAR_INCOME_ARCHIVED_FORMULA_OUTPUT_URL = (
    _ARCHIVED_ROOT + "/extended_cps.py#L837-L848"
)
PRIOR_YEAR_INCOME_ARCHIVED_FINALIZER_URL = (
    _ARCHIVED_ROOT + "/extended_cps.py#L1463-L1499"
)

US_PRIOR_YEAR_INCOME_STAGE_NAME = "prior_year_income"
US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS: tuple[str, ...] = (
    "employment_income_last_year",
    "self_employment_income_last_year",
    "previous_year_income_available",
)
US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "self_employment_income_last_year",
    "previous_year_income_available",
)
US_PRIOR_YEAR_INCOME_NONCONSTANT_PERSON_COLUMNS = (
    US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS
)
US_PRIOR_YEAR_INCOME_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "source_year",
    "PERIDNUM",
    "WSAL_VAL",
    "SEMP_VAL",
    "I_ERNVAL",
    "I_SEVAL",
)

_PERSON_WEIGHT_COLUMN = "person_weight"
_PERSON_SUPPORT_CHANNEL_COLUMN = "person_support_channel"
_PERSON_SUPPORT_SOURCE_ID_COLUMN = "person_source_id"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
_FORMULA_OWNED_OUTPUT = "employment_income_last_year"
_PUF_QRF_OUTPUT_COLUMNS: tuple[str, ...] = (
    "employment_income_last_year",
    "self_employment_income_last_year",
)
_PUF_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)
_PUF_PREDICTOR_PREFIX = "prior_year_income_predictor_"
_DERIVE_PARAMETER_KEYS = frozenset(
    {
        "person_id",
        "source_year",
        "prior_year_offset",
        "employment_source",
        "self_employment_source",
        "employment_allocation_flag",
        "self_employment_allocation_flag",
        "unallocated_flag",
        "sentinels",
        "fallback_to_current",
        "no_prior_artifact",
    }
)
_EXPECTED_DERIVE_PARAMETERS: dict[str, Any] = {
    "person_id": "PERIDNUM",
    "source_year": "source_year",
    "prior_year_offset": -1,
    "employment_source": "WSAL_VAL",
    "self_employment_source": "SEMP_VAL",
    "employment_allocation_flag": "I_ERNVAL",
    "self_employment_allocation_flag": "I_SEVAL",
    "unallocated_flag": 0,
    "sentinels": [-1, -9999],
    "fallback_to_current": True,
    "no_prior_artifact": "leave_defaults",
}
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {
        "predictors",
        "outputs",
        "max_train_samples",
        "n_estimators",
        "seed_from_build_config",
        "weight",
    }
)
_PREVIOUS_YEAR_AVAILABLE_SHARE_BAND = (0.05, 0.50)
_SELF_EMPLOYMENT_NONZERO_SHARE_BAND = (0.01, 0.25)


def us_prior_year_income_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged prior-year-income stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_PRIOR_YEAR_INCOME_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_PRIOR_YEAR_INCOME_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_PRIOR_YEAR_INCOME_STAGE_NAME]
    if tuple(spec.outputs) != US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS:
        raise ValueError(
            "prior_year_income manifest outputs do not match the runtime-owned family."
        )
    if "self_employment_income_last_year" in spec.nonnegative_outputs:
        raise ValueError(
            "self_employment_income_last_year must remain signed; the archived "
            "ASEC derivation preserves reported losses."
        )
    return spec


def _validated_derive_parameters(operation: SourceOperationSpec) -> None:
    unexpected = sorted(set(operation.parameters) - _DERIVE_PARAMETER_KEYS)
    missing = sorted(_DERIVE_PARAMETER_KEYS - set(operation.parameters))
    if unexpected or missing:
        raise SourceRuntimeError(
            "US prior-year-income derivation parameters drifted from the "
            f"archived method; missing={missing}, unexpected={unexpected}."
        )
    if dict(operation.parameters) != _EXPECTED_DERIVE_PARAMETERS:
        raise SourceRuntimeError(
            "US prior-year-income derivation parameters must exactly match "
            f"the archived method: expected {_EXPECTED_DERIVE_PARAMETERS}, "
            f"got {dict(operation.parameters)}."
        )


def _numeric_source(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def derive_us_prior_year_income_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Reproduce the archived adjacent-ASEC ``PERIDNUM`` earnings join."""

    if operation.kind != "derive_prior_year_income":
        raise SourceRuntimeError(
            "US prior-year-income derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US prior-year-income derivation requires the pooled person table."
        )
    _validated_derive_parameters(operation)
    missing = [
        column
        for column in US_PRIOR_YEAR_INCOME_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            "US prior-year-income derivation requires measured ASEC source "
            f"column(s): {missing}."
        )
    if _PERSON_SUPPORT_CHANNEL_COLUMN in frame.columns:
        absent = [
            column
            for column in US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS
            if column not in frame.columns
        ]
        if absent:
            raise SourceRuntimeError(
                "US prior-year income must be derived before support cloning; "
                f"the expanded frame is missing {absent}."
            )
        return frame.copy(deep=True)

    result = frame.copy(deep=True)
    source_year = pd.to_numeric(result["source_year"], errors="coerce")
    valid_year = np.isfinite(source_year.to_numpy(dtype=np.float64)) & (
        source_year.to_numpy(dtype=np.float64)
        == np.floor(source_year.to_numpy(dtype=np.float64))
    )
    if not valid_year.all():
        rows = np.flatnonzero(~valid_year)[:5].tolist()
        raise SourceRuntimeError(
            f"US prior-year-income source_year is invalid at row(s) {rows}."
        )
    if result["PERIDNUM"].isna().any():
        rows = result.index[result["PERIDNUM"].isna()].tolist()[:5]
        raise SourceRuntimeError(
            f"US prior-year-income PERIDNUM is missing at row(s) {rows}."
        )
    key = pd.MultiIndex.from_arrays(
        [source_year.astype("int64"), result["PERIDNUM"].astype(str)],
        names=["source_year", "PERIDNUM"],
    )
    duplicated = key.duplicated(keep=False)
    if duplicated.any():
        examples = list(dict.fromkeys(key[duplicated].tolist()))[:5]
        raise SourceRuntimeError(
            "US prior-year-income join requires unique (source_year, PERIDNUM) "
            f"keys; duplicate key(s): {examples}."
        )

    employment = _numeric_source(result, "WSAL_VAL")
    self_employment = _numeric_source(result, "SEMP_VAL")
    employment_flag = _numeric_source(result, "I_ERNVAL")
    self_employment_flag = _numeric_source(result, "I_SEVAL")
    eligible_prior = employment_flag.eq(0) & self_employment_flag.eq(0)

    prior = pd.DataFrame(
        {
            "source_year": source_year.loc[eligible_prior].astype("int64") + 1,
            "PERIDNUM": result.loc[eligible_prior, "PERIDNUM"].astype(str),
            "employment_income_last_year": employment.loc[eligible_prior],
            "self_employment_income_last_year": self_employment.loc[eligible_prior],
        }
    ).set_index(["source_year", "PERIDNUM"])
    current_key = pd.MultiIndex.from_arrays(
        [source_year.astype("int64"), result["PERIDNUM"].astype(str)],
        names=["source_year", "PERIDNUM"],
    )
    aligned = prior.reindex(current_key).reset_index(drop=True)

    sentinels = {-1.0, -9999.0}
    invalid_prior = aligned["employment_income_last_year"].isin(sentinels) | aligned[
        "self_employment_income_last_year"
    ].isin(sentinels)
    aligned.loc[
        invalid_prior,
        ["employment_income_last_year", "self_employment_income_last_year"],
    ] = np.nan
    available = (
        aligned["employment_income_last_year"].notna()
        & aligned["self_employment_income_last_year"].notna()
    )

    current_employment = employment.mask(employment.isin(sentinels))
    current_self_employment = self_employment.mask(self_employment.isin(sentinels))
    source_year_values = set(source_year.astype("int64").tolist())
    cohorts_with_prior_artifact = {prior_year + 1 for prior_year in source_year_values}
    can_fallback = source_year.astype("int64").isin(cohorts_with_prior_artifact)
    current_employment = current_employment.where(can_fallback)
    current_self_employment = current_self_employment.where(can_fallback)
    result["employment_income_last_year"] = (
        aligned["employment_income_last_year"]
        .fillna(current_employment.reset_index(drop=True))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    result["self_employment_income_last_year"] = (
        aligned["self_employment_income_last_year"]
        .fillna(current_self_employment.reset_index(drop=True))
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    result["previous_year_income_available"] = available.to_numpy(dtype=bool)
    if (result["employment_income_last_year"] < 0).any():
        rows = result.index[result["employment_income_last_year"] < 0].tolist()[:5]
        raise SourceRuntimeError(
            "US prior-year wage income must be nonnegative after sentinel "
            f"replacement; negative value(s) at row(s) {rows}."
        )
    return result


def impute_us_prior_year_income_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Jointly QRF-impute prior-year earnings on the PUF support half."""

    if operation.kind != "impute_prior_year_income_to_puf_support":
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation requires the person table."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation parameters must match the "
            f"archived method; missing={missing_parameters}, "
            f"unexpected={unexpected}."
        )
    if _PERSON_SUPPORT_CHANNEL_COLUMN not in frame.columns:
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    outputs = tuple(str(value) for value in operation.parameters["outputs"])
    if predictors != _PUF_PREDICTORS or outputs != _PUF_QRF_OUTPUT_COLUMNS:
        raise SourceRuntimeError(
            "US prior-year-income PUF predictors/outputs drifted from the "
            f"archived joint fit: predictors={list(predictors)}, "
            f"outputs={list(outputs)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation must use typed person weights."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation seed must come from the build."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError("US prior-year-income PUF fit sizes must be positive.")

    predictor_columns = [_PUF_PREDICTOR_PREFIX + value for value in predictors]
    required = [
        _PERSON_WEIGHT_COLUMN,
        *predictor_columns,
        *_PUF_QRF_OUTPUT_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation is missing source column(s): "
            f"{missing}."
        )

    channel = frame[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str)
    asec_mask = channel.eq(_BASE_ASEC_SUPPORT_CHANNEL)
    puf_mask = channel.eq(_PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation requires nonempty ASEC and "
            "PUF-tax-detail support channels."
        )

    training = frame.loc[
        asec_mask,
        [*predictor_columns, *_PUF_QRF_OUTPUT_COLUMNS],
    ].copy()
    training.columns = [*predictors, *_PUF_QRF_OUTPUT_COLUMNS]
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    )
    weight_values = weights.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(weight_values).all()
        or (weight_values < 0).any()
        or float(weight_values.sum()) <= 0
    ):
        raise SourceRuntimeError(
            "US prior-year-income QRF requires finite, nonnegative typed "
            "person weights with positive total mass."
        )
    if len(training) > max_train_samples:
        sample = training.sample(
            n=max_train_samples,
            random_state=(context.config.seed if context is not None else 0),
        ).index
        training = training.loc[sample]
        weights = weights.loc[sample]
        sampled_weight_values = weights.to_numpy(dtype=np.float64)
        if (
            not np.isfinite(sampled_weight_values).all()
            or (sampled_weight_values < 0).any()
            or float(sampled_weight_values.sum()) <= 0
        ):
            raise SourceRuntimeError(
                "US prior-year-income QRF sampled training weights must be "
                "finite and nonnegative with positive total mass."
            )

    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = list(predictors)
    for column in (*predictors, *_PUF_QRF_OUTPUT_COLUMNS):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US prior-year-income QRF training column {column!r} "
                "contains nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US prior-year-income QRF prediction column {column!r} "
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
        list(_PUF_QRF_OUTPUT_COLUMNS),
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test)
    if not isinstance(predictions, pd.DataFrame):
        raise SourceRuntimeError(
            "US prior-year-income QRF must return a pandas DataFrame."
        )
    missing_predictions = [
        column for column in _PUF_QRF_OUTPUT_COLUMNS if column not in predictions
    ]
    if missing_predictions:
        raise SourceRuntimeError(
            f"US prior-year-income QRF omitted output column(s): {missing_predictions}."
        )
    expected_predictions = int(puf_mask.sum())
    if len(predictions) != expected_predictions:
        raise SourceRuntimeError(
            "US prior-year-income QRF returned "
            f"{len(predictions)} rows; expected {expected_predictions}."
        )
    for column in _PUF_QRF_OUTPUT_COLUMNS:
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise SourceRuntimeError(
                f"US prior-year-income QRF output {column!r} contains nonfinite values."
            )
        if column == _FORMULA_OWNED_OUTPUT and (values < 0).any():
            rows = np.flatnonzero(values < 0)[:5].tolist()
            raise SourceRuntimeError(
                "US prior-year-income QRF predicted negative wage income at "
                f"PUF prediction row(s) {rows}."
            )

    result = frame.copy(deep=True)
    for column in _PUF_QRF_OUTPUT_COLUMNS:
        result.loc[puf_mask, column] = pd.to_numeric(
            predictions[column], errors="coerce"
        ).to_numpy(dtype=np.float64)
    return result


def _person_prior_year_income_predictors(frame: Frame) -> pd.DataFrame:
    """Build the archived eight second-stage predictors on person rows."""

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
                        "US prior-year-income QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation cannot construct a predictor "
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
            "US prior-year-income PUF imputation requires is_male, is_female, "
            "or measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError(
            "US prior-year-income PUF imputation requires has_esi."
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
            "US prior-year-income PUF imputation requires tax-unit filing status."
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
            "US prior-year-income PUF imputation requires tax_unit_role_input."
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
    predictors["tax_unit_count_dependents"] = dependent.groupby(
        person["person_tax_unit_id"]
    ).transform("sum")
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


def _replace_person_table(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def with_us_prior_year_income_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
) -> Frame:
    """Materialize adjacent-year earnings and PUF-support replacements."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US prior-year income requires the US schema.")
    person = frame.table("person")
    has_support_channels = _PERSON_SUPPORT_CHANNEL_COLUMN in person.columns
    has_raw_sources = all(
        column in person for column in US_PRIOR_YEAR_INCOME_REQUIRED_SOURCE_COLUMNS
    )
    if (
        not has_support_channels
        and all(column in person for column in US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS)
        and not has_raw_sources
    ):
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_channels:
        predictors = _person_prior_year_income_predictors(frame)
        for column in _PUF_PREDICTORS:
            stage_person[_PUF_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_prior_year_income_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_prior_year_income": derive_us_prior_year_income_from_manifest,
            "impute_prior_year_income_to_puf_support": (
                impute_us_prior_year_income_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=seed, target_year=time_period),
    ).drop(
        columns=[
            _PERSON_WEIGHT_COLUMN,
            *(_PUF_PREDICTOR_PREFIX + value for value in _PUF_PREDICTORS),
        ],
        errors="ignore",
    )
    if has_support_channels:
        output = output.drop(columns=[_FORMULA_OWNED_OUTPUT])
    return _replace_person_table(frame, output)


def us_prior_year_income_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal, signed mass, and support-channel diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    availability = person["previous_year_income_available"].astype(bool).to_numpy()
    self_employment = pd.to_numeric(
        person["self_employment_income_last_year"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    nonzero = self_employment != 0

    def _share(mask: np.ndarray) -> float:
        return float(weights[mask].sum()) / total_weight if total_weight > 0 else 0.0

    channels: dict[str, dict[str, float | int]] = {}
    if _PERSON_SUPPORT_CHANNEL_COLUMN in person:
        support_channel = person[_PERSON_SUPPORT_CHANNEL_COLUMN].astype(str).to_numpy()
        for channel in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL):
            mask = support_channel == channel
            channel_weight = float(weights[mask].sum())
            channels[channel] = {
                "rows": int(mask.sum()),
                "weighted_population": channel_weight,
                "availability_share": (
                    float(weights[mask & availability].sum()) / channel_weight
                    if channel_weight > 0
                    else 0.0
                ),
                "self_employment_nonzero_share": (
                    float(weights[mask & nonzero].sum()) / channel_weight
                    if channel_weight > 0
                    else 0.0
                ),
                "self_employment_negative_rows": int(
                    np.count_nonzero(mask & (self_employment < 0))
                ),
            }

    clone_availability_mismatches = 0
    if _PERSON_SUPPORT_SOURCE_ID_COLUMN in person:
        clone_availability_mismatches = int(
            person.assign(_availability=availability)
            .groupby(_PERSON_SUPPORT_SOURCE_ID_COLUMN, sort=False)["_availability"]
            .nunique()
            .gt(1)
            .sum()
        )

    return {
        "rows": int(len(person)),
        "weighted_population": total_weight,
        "previous_year_income_available_share": _share(availability),
        "previous_year_income_available_share_band": list(
            _PREVIOUS_YEAR_AVAILABLE_SHARE_BAND
        ),
        "self_employment_income_last_year_nonzero_share": _share(nonzero),
        "self_employment_income_last_year_nonzero_share_band": list(
            _SELF_EMPLOYMENT_NONZERO_SHARE_BAND
        ),
        "self_employment_income_last_year_positive_rows": int(
            np.count_nonzero(self_employment > 0)
        ),
        "self_employment_income_last_year_negative_rows": int(
            np.count_nonzero(self_employment < 0)
        ),
        "self_employment_income_last_year_weighted_total": float(
            np.dot(weights, self_employment)
        ),
        "unique_counts": {
            column: int(person[column].dropna().nunique())
            for column in US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS
        },
        "channels": channels,
        "clone_availability_mismatches": clone_availability_mismatches,
    }


def us_prior_year_income_signal_gate(frame: Frame) -> GateResult:
    """Fail closed on missing, default-only, or implausible prior-year inputs."""

    person = frame.table("person")
    missing = [
        column
        for column in US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS
        if column not in person
    ]
    if missing:
        return GateResult(
            name="prior_year_income_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )
    availability_source = person["previous_year_income_available"]
    if availability_source.isna().any() or not pd.api.types.is_bool_dtype(
        availability_source.dtype
    ):
        return GateResult(
            name="prior_year_income_signal",
            passed=False,
            failures=(
                "previous_year_income_available must be a complete boolean "
                "source column.",
            ),
            details={
                "dtype": str(availability_source.dtype),
                "missing": int(availability_source.isna().sum()),
            },
        )
    self_employment = pd.to_numeric(
        person["self_employment_income_last_year"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(self_employment).all():
        rows = np.flatnonzero(~np.isfinite(self_employment))[:5].tolist()
        return GateResult(
            name="prior_year_income_signal",
            passed=False,
            failures=(
                "self_employment_income_last_year contains nonfinite value(s) "
                f"at row(s) {rows}.",
            ),
            details={"nonfinite_rows": rows},
        )

    summary = us_prior_year_income_summary(frame)
    failures: list[str] = []
    checks = (
        (
            "previous_year_income_available_share",
            "previous_year_income_available_share_band",
            "previous-year availability weighted share",
        ),
        (
            "self_employment_income_last_year_nonzero_share",
            "self_employment_income_last_year_nonzero_share_band",
            "prior-year self-employment nonzero weighted share",
        ),
    )
    for share_key, band_key, label in checks:
        share = float(summary[share_key])
        lower, upper = summary[band_key]
        if not lower <= share <= upper:
            failures.append(f"{label} {share:.6f} outside [{lower:.6f}, {upper:.6f}].")
    for column, count in summary["unique_counts"].items():
        if int(count) < 2:
            failures.append(f"{column} is degenerate with {count} distinct value(s).")
    channels = summary["channels"]
    if channels:
        for channel in (_BASE_ASEC_SUPPORT_CHANNEL, _PUF_TAX_DETAIL_SUPPORT_CHANNEL):
            channel_summary = channels.get(channel)
            if not channel_summary or int(channel_summary["rows"]) == 0:
                failures.append(f"{channel} prior-year-income support is empty.")
                continue
            if float(channel_summary["self_employment_nonzero_share"]) <= 0:
                failures.append(
                    f"{channel} self_employment_income_last_year is default-only."
                )
            if float(channel_summary["availability_share"]) <= 0:
                failures.append(
                    f"{channel} previous_year_income_available is default-only."
                )
        asec = channels.get(_BASE_ASEC_SUPPORT_CHANNEL)
        if asec and int(asec["self_employment_negative_rows"]) == 0:
            failures.append(
                "ASEC self_employment_income_last_year lost all signed loss signal."
            )
    elif int(summary["self_employment_income_last_year_negative_rows"]) == 0:
        failures.append("self_employment_income_last_year lost all signed loss signal.")
    mismatches = int(summary["clone_availability_mismatches"])
    if mismatches:
        failures.append(
            f"{mismatches} source person(s) disagree on prior-year availability "
            "across support clones."
        )
    return GateResult(
        name="prior_year_income_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def us_prior_year_income_source_reconciliation_gate(frame: Frame) -> GateResult:
    """Recompute the ASEC carry and require exact persisted-value agreement.

    This runs on the reusable base before sparse support selection, while all
    adjacent-year source rows are still present.  Only the ASEC channel is
    reconciled: PUF-support amounts are deliberately owned by the joint QRF.
    """

    person = frame.table("person")
    if _PERSON_SUPPORT_CHANNEL_COLUMN in person:
        asec_mask = (
            person[_PERSON_SUPPORT_CHANNEL_COLUMN]
            .astype(str)
            .eq(_BASE_ASEC_SUPPORT_CHANNEL)
        )
        actual = person.loc[asec_mask].reset_index(drop=True).copy()
    else:
        actual = person.reset_index(drop=True).copy()
    missing_sources = [
        column
        for column in US_PRIOR_YEAR_INCOME_REQUIRED_SOURCE_COLUMNS
        if column not in actual
    ]
    compare_columns = [
        column for column in US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS if column in actual
    ]
    missing_outputs = sorted(
        set(US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS) - set(compare_columns)
    )
    if missing_sources or missing_outputs:
        failures = []
        if missing_sources:
            failures.append(f"ASEC source columns missing: {missing_sources}.")
        if missing_outputs:
            failures.append(f"ASEC persisted outputs missing: {missing_outputs}.")
        return GateResult(
            name="prior_year_income_source_reconciliation",
            passed=False,
            failures=tuple(failures),
            details={
                "missing_sources": missing_sources,
                "missing_outputs": missing_outputs,
            },
        )

    source = actual.drop(
        columns=[
            *US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS,
            _PERSON_SUPPORT_CHANNEL_COLUMN,
        ],
        errors="ignore",
    )
    operation = next(
        operation
        for operation in us_prior_year_income_stage_spec().operations
        if operation.kind == "derive_prior_year_income"
    )
    expected = derive_us_prior_year_income_from_manifest(source, operation, None)
    mismatch_counts: dict[str, int] = {}
    for column in compare_columns:
        if column == "previous_year_income_available":
            mismatches = actual[column].to_numpy(dtype=bool) != expected[
                column
            ].to_numpy(dtype=bool)
        else:
            observed = pd.to_numeric(actual[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            target = expected[column].to_numpy(dtype=np.float64)
            mismatches = ~np.isclose(observed, target, rtol=0.0, atol=0.0)
        mismatch_counts[column] = int(np.count_nonzero(mismatches))
    failures = tuple(
        f"ASEC {column} differs from the archived adjacent-year derivation on "
        f"{count} row(s)."
        for column, count in mismatch_counts.items()
        if count
    )
    return GateResult(
        name="prior_year_income_source_reconciliation",
        passed=not failures,
        failures=failures,
        details={
            "asec_rows": int(len(actual)),
            "compared_columns": compare_columns,
            "mismatch_counts": mismatch_counts,
        },
    )
