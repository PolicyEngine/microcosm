"""ASEC-reported retirement contributions split into PolicyEngine leaves.

The retired eCPS pipeline read the measured annual contribution total from
ASEC ``RETCB_VAL`` and allocated it across five account-type inputs.  The
allocation is pinned to the final archived implementation at
``9a823603e6b5fb916d65ec45d74c9c7eb0043db1``:

* ``datasets/cps/cps.py`` lines 1504-1552 gates the self-employed share on
  positive ``SEMP_VAL``, the defined-contribution share on positive
  ``WSAL_VAL``, and the IRA remainder on either earnings source; and
* ``datasets/cps/imputation_parameters.yaml`` lines 24-55 records the
  administrative sources and the four allocation shares.

The outputs are the five ``*_desired`` input leaves.  PolicyEngine-US 1.819.0
applies the statutory combined elective-deferral and IRA limits and the
self-employed-plan cap, so this stage preserves the uncapped measured desired
amounts.
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
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from microcosm.build.us_runtime.support_provenance import (
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_gate_source_channel_series,
    support_role_series,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_RETIREMENT_CONTRIBUTION_NONCONSTANT_PERSON_COLUMNS",
    "US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS",
    "US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS",
    "US_RETIREMENT_CONTRIBUTION_STAGE_NAME",
    "derive_us_retirement_contributions_from_manifest",
    "impute_us_retirement_contributions_to_puf_support_from_manifest",
    "us_retirement_contributions_signal_gate",
    "us_retirement_contributions_stage_spec",
    "us_retirement_contributions_summary",
    "with_us_retirement_contribution_inputs",
]

QRF: Any | None = None

US_RETIREMENT_CONTRIBUTION_STAGE_NAME = "retirement_contributions"

US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "traditional_401k_contributions_desired",
    "roth_401k_contributions_desired",
    "traditional_ira_contributions_desired",
    "roth_ira_contributions_desired",
    "self_employed_pension_contributions_desired",
)
US_RETIREMENT_CONTRIBUTION_NONCONSTANT_PERSON_COLUMNS = (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
)
US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "RETCB_VAL",
    "WSAL_VAL",
    "SEMP_VAL",
)

_PERSON_WEIGHT_COLUMN = "person_weight"
_BASE_ASEC_SUPPORT_CHANNEL = "asec"
_PUF_TAX_DETAIL_SUPPORT_CHANNEL = "puf_tax_detail"
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
_PUF_PREDICTOR_PREFIX = "retirement_contribution_predictor_"
_SHARE_PARAMETER_KEYS = frozenset(
    {
        "se_pension_share",
        "dc_share_of_remainder",
        "roth_dc_share",
        "traditional_ira_share",
    }
)
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {
        "predictors",
        "max_train_samples",
        "n_estimators",
        "seed_from_build_config",
        "weight",
    }
)
# Deliberately broad source-plausibility bound.  Its job is to reject a
# default/near-universal surface, while the exact allocation identities below
# protect the family semantics.
_NONZERO_SHARE_BAND = (0.0001, 0.75)


def us_retirement_contributions_stage_spec() -> SourceStageSpec:
    """Load the packaged retirement-contribution source-stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_RETIREMENT_CONTRIBUTION_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_RETIREMENT_CONTRIBUTION_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_RETIREMENT_CONTRIBUTION_STAGE_NAME]
    missing = sorted(set(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_RETIREMENT_CONTRIBUTION_STAGE_NAME!r} manifest stage does not "
            f"declare output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _share_parameters(operation: SourceOperationSpec) -> dict[str, float]:
    unexpected = sorted(set(operation.parameters) - _SHARE_PARAMETER_KEYS)
    missing = sorted(_SHARE_PARAMETER_KEYS - set(operation.parameters))
    if unexpected or missing:
        raise SourceRuntimeError(
            "US retirement-contribution derivation parameters must match the "
            f"archived allocation contract; missing={missing}, "
            f"unexpected={unexpected}."
        )
    shares = {key: float(operation.parameters[key]) for key in _SHARE_PARAMETER_KEYS}
    invalid = {key: value for key, value in shares.items() if not 0 <= value <= 1}
    if invalid:
        raise SourceRuntimeError(
            "US retirement-contribution allocation shares must be in [0, 1]: "
            f"{invalid}."
        )
    return shares


def _numeric_source(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US retirement-contribution source {column!r} contains "
            f"{nonfinite} nonfinite value(s); the measured source must not be "
            "silently replaced."
        )
    return values


def _asec_source_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select physical ASEC rows while retaining the legacy all-row source."""

    if not has_assembled_support_metadata(frame, entity="person"):
        return np.ones(len(frame), dtype=bool)
    source_channels = support_gate_source_channel_series(frame, entity="person")
    mask = source_channels.eq(_BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
    if not mask.any():
        raise SourceRuntimeError(
            "US retirement-contribution support has no physical ASEC source rows."
        )
    return mask


def _source_reconciliation_mask(frame: pd.DataFrame) -> np.ndarray:
    """Select direct ASEC operator rows whose allocation remains source-exact."""

    source_mask = _asec_source_mask(frame)
    if not has_assembled_support_metadata(frame, entity="person"):
        return source_mask
    roles = support_role_series(frame, entity="person").to_numpy()
    return source_mask & (roles == _BASE_ASEC_SUPPORT_CHANNEL)


def derive_us_retirement_contributions_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Split measured ``RETCB_VAL`` across the five desired input leaves."""

    if operation.kind != "derive_retirement_contributions":
        raise SourceRuntimeError(
            "US retirement-contribution derivation received unexpected "
            f"operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US retirement-contribution derivation requires the person table "
            "to be read first."
        )
    missing = [
        column
        for column in US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            "US retirement-contribution derivation requires measured ASEC "
            f"source column(s): {missing}."
        )
    shares = _share_parameters(operation)

    result = frame.copy(deep=True)
    retirement_contributions = _numeric_source(result, "RETCB_VAL")
    negative_source = int(np.count_nonzero(retirement_contributions < 0))
    if negative_source:
        raise SourceRuntimeError(
            "US retirement-contribution source 'RETCB_VAL' contains "
            f"{negative_source} negative value(s)."
        )
    has_wages = _numeric_source(result, "WSAL_VAL") > 0
    has_self_employment = _numeric_source(result, "SEMP_VAL") > 0
    has_earned_income = has_wages | has_self_employment

    self_employed = np.where(
        has_self_employment,
        retirement_contributions * shares["se_pension_share"],
        0.0,
    )
    remaining = np.maximum(retirement_contributions - self_employed, 0.0)
    dc_pool = np.where(
        has_wages,
        remaining * shares["dc_share_of_remainder"],
        0.0,
    )
    ira_pool = np.where(has_earned_income, remaining - dc_pool, 0.0)

    result["traditional_401k_contributions_desired"] = dc_pool * (
        1.0 - shares["roth_dc_share"]
    )
    result["roth_401k_contributions_desired"] = dc_pool * shares["roth_dc_share"]
    result["traditional_ira_contributions_desired"] = (
        ira_pool * shares["traditional_ira_share"]
    )
    result["roth_ira_contributions_desired"] = ira_pool * (
        1.0 - shares["traditional_ira_share"]
    )
    result["self_employed_pension_contributions_desired"] = self_employed
    return result


def impute_us_retirement_contributions_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """QRF-impute all five contributions onto the PUF support channel.

    This is the retired second-stage treatment: train on the measured ASEC
    rows after their direct split, condition on the clone's PUF-imputed income,
    replace all five PUF-half values, clip nonnegative, and zero employer-plan
    contributions without wages and self-employed-plan contributions without
    self-employment income.

    A non-expanded ASEC frame has no support-channel column; in that first pass
    this operation is intentionally a no-op.  The base builder runs the same
    manifest again after PUF tax-detail imputation, when both support channels
    exist and this operation becomes active.
    """

    if operation.kind != "impute_retirement_contributions_to_puf_support":
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation received unexpected "
            f"operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation requires the person "
            "table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation parameters must match "
            f"the archived method; missing={missing_parameters}, "
            f"unexpected={unexpected}."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PUF_PREDICTORS:
        raise SourceRuntimeError(
            "US retirement-contribution PUF predictors drifted from the "
            f"archived method: expected {list(_PUF_PREDICTORS)}, got "
            f"{list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation must use the typed "
            f"person-weight column {_PERSON_WEIGHT_COLUMN!r}."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation seed must come from "
            "the build config."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError(
            "US retirement-contribution PUF max_train_samples and n_estimators "
            "must be positive."
        )

    required = [
        _PERSON_WEIGHT_COLUMN,
        *(_PUF_PREDICTOR_PREFIX + predictor for predictor in predictors),
        *US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation is missing source "
            f"column(s): {missing}."
        )

    roles = support_role_series(frame, entity="person")
    asec_mask = roles == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = roles == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation requires nonempty ASEC "
            "and PUF-tax-detail support channels."
        )

    predictor_columns = [_PUF_PREDICTOR_PREFIX + predictor for predictor in predictors]
    training_columns = [
        *predictor_columns,
        *US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    ]
    training = frame.loc[asec_mask, training_columns].copy()
    training.columns = [
        *(predictors),
        *US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    ]
    weights = pd.to_numeric(
        frame.loc[asec_mask, _PERSON_WEIGHT_COLUMN], errors="coerce"
    ).fillna(0.0)
    if len(training) > max_train_samples:
        sample = training.sample(
            n=max_train_samples,
            random_state=(context.config.seed if context is not None else 0),
        ).index
        training = training.loc[sample]
        weights = weights.loc[sample]

    test = frame.loc[puf_mask, predictor_columns].copy()
    test.columns = list(predictors)
    for column in (*predictors, *US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS):
        if column in training:
            training[column] = pd.to_numeric(training[column], errors="coerce")
            if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
                raise SourceRuntimeError(
                    "US retirement-contribution QRF training column "
                    f"{column!r} contains nonfinite values."
                )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                "US retirement-contribution QRF prediction column "
                f"{column!r} contains nonfinite values."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("microcosm.fit").QRF
    seed = context.config.seed if context is not None else 0
    fitted = QRF(n_estimators=n_estimators, seed=seed).fit(
        training,
        list(predictors),
        list(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS),
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test).clip(lower=0.0)
    no_wages = test["employment_income"].to_numpy(dtype=np.float64) == 0
    no_self_employment = test["self_employment_income"].to_numpy(dtype=np.float64) == 0
    predictions.loc[
        no_wages,
        [
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
        ],
    ] = 0.0
    predictions.loc[
        no_self_employment,
        "self_employed_pension_contributions_desired",
    ] = 0.0

    result = frame.copy(deep=True)
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        result.loc[puf_mask, column] = predictions[column].to_numpy(dtype=np.float64)
    return result


def _person_retirement_predictors(frame: Frame) -> pd.DataFrame:
    """Build the retired QRF predictors on person rows, failing closed."""

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
                        "US retirement-contribution QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation cannot construct "
            f"predictor from any of {list(columns)}."
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
            "US retirement-contribution PUF imputation requires is_male, "
            "is_female, or measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError(
            "US retirement-contribution PUF imputation requires has_esi."
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
            "US retirement-contribution PUF imputation requires tax-unit filing status."
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
            "US retirement-contribution PUF imputation requires "
            "tax_unit_role_input to count dependents."
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
    dependent_count = dependent.groupby(person["person_tax_unit_id"]).transform("sum")
    predictors["tax_unit_count_dependents"] = dependent_count.to_numpy(dtype=np.float64)
    predictors["employment_income"] = _numeric(
        "employment_income_before_lsr",
        "WSAL_VAL",
    )
    predictors["self_employment_income"] = _numeric(
        "self_employment_income_before_lsr",
        "SEMP_VAL",
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


def with_us_retirement_contribution_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
) -> Frame:
    """Materialize the five retirement-contribution inputs on a US frame."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US retirement contributions require the US schema.")
    person = frame.table("person")
    has_support_roles = has_support_role_metadata(person, entity="person")
    if _retirement_contribution_surface_carries_signal(frame) and not has_support_roles:
        return frame

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_roles:
        predictors = _person_retirement_predictors(frame)
        for column in _PUF_PREDICTORS:
            stage_person[_PUF_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_retirement_contributions_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_retirement_contributions": (
                derive_us_retirement_contributions_from_manifest
            ),
            "impute_retirement_contributions_to_puf_support": (
                impute_us_retirement_contributions_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                "US retirement-contribution stage output does not cover every "
                f"person for {column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_retirement_contributions_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and source-allocation diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    contributions = {
        column: pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
    }
    source = np.full(len(person), np.nan, dtype=np.float64)
    source_mask = np.zeros(len(person), dtype=bool)
    reconciliation_mask = np.zeros(len(person), dtype=bool)
    if "RETCB_VAL" in person:
        source_mask = _asec_source_mask(person)
        source[source_mask] = _numeric_source(
            person.loc[source_mask],
            "RETCB_VAL",
        )
        reconciliation_mask = _source_reconciliation_mask(person)
    combined = np.sum(np.column_stack(tuple(contributions.values())), axis=1)

    def _share(values: np.ndarray) -> float:
        positive = np.isfinite(values) & (values > 0)
        return (
            float(weights[positive].sum()) / total_weight if total_weight > 0 else 0.0
        )

    nonfinite = {
        column: int(np.count_nonzero(~np.isfinite(values)))
        for column, values in contributions.items()
    }
    negative = {
        column: int(np.count_nonzero(values < 0))
        for column, values in contributions.items()
    }
    source_positive = reconciliation_mask & (source > 0)
    allocation_mismatch = source_positive & ~np.isclose(
        combined,
        source,
        rtol=1e-9,
        atol=1e-6,
    )
    return {
        "nonzero_shares": {
            column: _share(values) for column, values in contributions.items()
        },
        "weighted_totals": {
            column: float(np.sum(np.nan_to_num(values) * weights))
            for column, values in contributions.items()
        },
        "nonzero_share_band": list(_NONZERO_SHARE_BAND),
        "nonfinite": nonfinite,
        "negative": negative,
        "source_rows": int(np.count_nonzero(source_mask)),
        "source_reconciliation_rows": int(np.count_nonzero(reconciliation_mask)),
        "source_positive_rows": int(np.count_nonzero(source_positive)),
        "allocation_mismatch_rows": int(np.count_nonzero(allocation_mismatch)),
        "source_total": float(
            np.sum(np.where(reconciliation_mask, source, 0.0) * weights)
        ),
        "allocated_total": float(np.sum(np.nan_to_num(combined) * weights)),
    }


def us_retirement_contributions_signal_gate(frame: Frame) -> GateResult:
    """Require every retirement-contribution leaf to carry valid signal."""

    person = frame.table("person")
    missing = [
        column
        for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="retirement_contributions_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_retirement_contributions_summary(frame)
    failures: list[str] = []
    low, high = summary["nonzero_share_band"]
    for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS:
        nonfinite = int(summary["nonfinite"][column])
        negative = int(summary["negative"][column])
        share = float(summary["nonzero_shares"][column])
        if nonfinite:
            failures.append(f"{column}: {nonfinite} nonfinite values.")
        if negative:
            failures.append(f"{column}: {negative} negative values.")
        if not (low <= share <= high):
            failures.append(
                f"{column}: nonzero share {share:.4f} outside plausibility "
                f"band [{low}, {high}]."
            )
    return GateResult(
        name="retirement_contributions_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _retirement_contribution_surface_carries_signal(frame: Frame) -> bool:
    person = frame.table("person")
    if not all(
        column in person for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
    ):
        return False
    return us_retirement_contributions_signal_gate(frame).passed
