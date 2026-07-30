"""ASEC-measured childcare expenses with retired PUF-half QRF treatment.

The retired eCPS pipeline carried measured annual SPM-unit childcare expenses
from ASEC ``SPM_CHILDCAREXPNS`` and then QRF-imputed that CPS-only input onto
the PUF clone half. The QRF used eight person predictors, at most 5,000 ASEC
training rows, and reduced non-person outputs with ``value_from_first_person``.
Immutable archived source coordinates are exposed below.

The ASEC value is replicated on every member of an SPM unit. This port refuses
missing, nonfinite, negative, or internally inconsistent replicas rather than
silently choosing a maximum or filling zeros. It also uses typed person weights
for the QRF fit: the archived call omitted weights, but weighted fitting is a
hard Populace build contract. PolicyEngine-US owns all downstream CDCC and SPM
formulas; this stage persists only the factual SPM-unit input.
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
    "CHILDCARE_ARCHIVED_CPS_DERIVATION_URL",
    "CHILDCARE_ARCHIVED_PUF_IMPUTATION_URL",
    "US_CHILDCARE_OUTPUT_COLUMNS",
    "US_CHILDCARE_REQUIRED_SOURCE_COLUMNS",
    "US_CHILDCARE_STAGE_NAME",
    "derive_us_childcare_from_manifest",
    "impute_us_childcare_to_puf_support_from_manifest",
    "us_childcare_signal_gate",
    "us_childcare_stage_spec",
    "us_childcare_summary",
    "with_us_childcare_inputs",
]

QRF: Any | None = None

_ARCHIVED_DATA_REPOSITORY = "policyengine-" + "us-data"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    f"{_ARCHIVED_DATA_REPOSITORY}/blob/"
    "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe/"
    "policyengine_" + "us_data/"
)
CHILDCARE_ARCHIVED_CPS_DERIVATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/cps.py#L1612-L1622"
)
CHILDCARE_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "datasets/cps/extended_cps.py#L639-L739"
)

US_CHILDCARE_STAGE_NAME = "childcare_inputs"
US_CHILDCARE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "spm_unit_pre_subsidy_childcare_expenses",
)
US_CHILDCARE_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_spm_unit_id",
    "SPM_CHILDCAREXPNS",
)

_OUTPUT = US_CHILDCARE_OUTPUT_COLUMNS[0]
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
_PUF_PREDICTOR_PREFIX = "childcare_predictor_"
_DIRECT_PARAMETER_KEYS = frozenset()
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {
        "predictors",
        "max_train_samples",
        "n_estimators",
        "seed_from_build_config",
        "weight",
        "reduction",
    }
)
_NONZERO_SHARE_BAND = (0.005, 0.50)
_REPLICA_ATOL = 1e-6


def us_childcare_stage_spec() -> SourceStageSpec:
    """Load the packaged childcare source-stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_CHILDCARE_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_CHILDCARE_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_CHILDCARE_STAGE_NAME]
    missing = sorted(set(US_CHILDCARE_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_CHILDCARE_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _numeric_source(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(values)))
    if nonfinite:
        raise SourceRuntimeError(
            f"US childcare source {column!r} contains {nonfinite} nonfinite "
            "value(s); the measured source must not be silently replaced."
        )
    return values


def derive_us_childcare_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Carry measured, replicated ASEC childcare values on person rows."""

    if operation.kind != "derive_childcare_inputs":
        raise SourceRuntimeError(
            f"US childcare derivation received unexpected operation {operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US childcare derivation requires the person table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _DIRECT_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            f"US childcare derivation received unsupported parameter(s): {unexpected}."
        )
    missing = [
        column
        for column in US_CHILDCARE_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            "US childcare derivation requires measured ASEC source column(s): "
            f"{missing}."
        )

    values = _numeric_source(frame, "SPM_CHILDCAREXPNS")
    negative = int(np.count_nonzero(values < 0.0))
    if negative:
        raise SourceRuntimeError(
            "US childcare source 'SPM_CHILDCAREXPNS' contains "
            f"{negative} negative value(s)."
        )
    spm_ids = pd.to_numeric(frame["person_spm_unit_id"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(spm_ids).all():
        raise SourceRuntimeError(
            "US childcare source person_spm_unit_id contains nonfinite values."
        )
    replicas = pd.DataFrame({"spm_unit_id": spm_ids, "value": values})
    bounds = replicas.groupby("spm_unit_id", sort=False)["value"].agg(["min", "max"])
    inconsistent = ~np.isclose(
        bounds["min"].to_numpy(dtype=np.float64),
        bounds["max"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=_REPLICA_ATOL,
    )
    if bool(inconsistent.any()):
        bad_ids = bounds.index[inconsistent].astype("int64").tolist()
        raise SourceRuntimeError(
            "US childcare source SPM_CHILDCAREXPNS disagrees within replicated "
            f"SPM unit(s): {bad_ids[:10]}."
        )

    result = frame.copy(deep=True)
    result[_OUTPUT] = values
    return result


def impute_us_childcare_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """QRF-impute childcare onto PUF people before SPM-unit reduction."""

    if operation.kind != "impute_childcare_to_puf_support":
        raise SourceRuntimeError(
            "US childcare PUF imputation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US childcare PUF imputation requires the person table to be read first."
        )
    unexpected = sorted(set(operation.parameters) - _PUF_IMPUTATION_PARAMETER_KEYS)
    missing_parameters = sorted(
        _PUF_IMPUTATION_PARAMETER_KEYS - set(operation.parameters)
    )
    if unexpected or missing_parameters:
        raise SourceRuntimeError(
            "US childcare PUF imputation parameters must match the archived "
            f"method; missing={missing_parameters}, unexpected={unexpected}."
        )
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)

    predictors = tuple(str(value) for value in operation.parameters["predictors"])
    if predictors != _PUF_PREDICTORS:
        raise SourceRuntimeError(
            "US childcare PUF predictors drifted from the archived method: "
            f"expected {list(_PUF_PREDICTORS)}, got {list(predictors)}."
        )
    if operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN:
        raise SourceRuntimeError(
            "US childcare PUF imputation must use the typed person-weight "
            f"column {_PERSON_WEIGHT_COLUMN!r}."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError(
            "US childcare PUF imputation seed must come from the build config."
        )
    if operation.parameters["reduction"] != "value_from_first_person":
        raise SourceRuntimeError(
            "US childcare PUF imputation must use the archived "
            "value_from_first_person reduction."
        )
    max_train_samples = int(operation.parameters["max_train_samples"])
    n_estimators = int(operation.parameters["n_estimators"])
    if max_train_samples <= 0 or n_estimators <= 0:
        raise SourceRuntimeError(
            "US childcare PUF max_train_samples and n_estimators must be positive."
        )

    predictor_columns = [_PUF_PREDICTOR_PREFIX + name for name in predictors]
    required = [_PERSON_WEIGHT_COLUMN, *predictor_columns, _OUTPUT]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceRuntimeError(
            f"US childcare PUF imputation is missing source column(s): {missing}."
        )

    role = support_role_series(frame, entity="person")
    asec_mask = role == _BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = role == _PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise SourceRuntimeError(
            "US childcare PUF imputation requires nonempty ASEC and "
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
            "US childcare QRF person weights must be finite and nonnegative."
        )
    if float(numeric_weights.sum()) <= 0.0:
        raise SourceRuntimeError("US childcare QRF person weights sum to zero.")

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
                f"US childcare QRF training column {column!r} contains "
                "nonfinite values."
            )
    for column in predictors:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise SourceRuntimeError(
                f"US childcare QRF prediction column {column!r} contains "
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
        [_OUTPUT],
        weights=weights.to_numpy(dtype=np.float64),
    )
    predictions = fitted.predict(test)
    if _OUTPUT not in predictions:
        raise SourceRuntimeError(
            f"US childcare QRF prediction is missing output {_OUTPUT!r}."
        )
    predicted = pd.to_numeric(predictions[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(predicted).all():
        raise SourceRuntimeError("US childcare QRF produced nonfinite predictions.")
    if bool((predicted < 0.0).any()):
        raise SourceRuntimeError("US childcare QRF produced negative predictions.")

    result = frame.copy(deep=True)
    result.loc[puf_mask, _OUTPUT] = predicted
    return result


def _person_childcare_predictors(frame: Frame) -> pd.DataFrame:
    """Build the retired eight QRF predictors on person rows."""

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
                        "US childcare QRF predictor source "
                        f"{column!r} contains nonfinite values."
                    )
                return values
        raise SourceRuntimeError(
            "US childcare PUF imputation cannot construct predictor from any "
            f"of {list(columns)}."
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
            "US childcare PUF imputation requires is_male, is_female, or "
            "measured A_SEX."
        )
    if "has_esi" not in person:
        raise SourceRuntimeError("US childcare PUF imputation requires has_esi.")
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
            "US childcare PUF imputation requires tax-unit filing status."
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
            "US childcare PUF imputation requires tax_unit_role_input to count "
            "dependents."
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


def with_us_childcare_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    allow_existing_without_source: bool = False,
) -> Frame:
    """Materialize measured/QRF childcare on the SPM-unit input leaf.

    ``allow_existing_without_source`` is only for a downstream release build
    consuming a base artifact that already passed this stage's signal gate.
    Source/base construction remains strict and cannot reuse clone duplicates.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US childcare inputs require the US schema.")
    person = frame.table("person")
    spm_unit = frame.table("spm_unit")
    source_available = all(
        column in person for column in US_CHILDCARE_REQUIRED_SOURCE_COLUMNS
    )
    if not source_available:
        if allow_existing_without_source and _childcare_surface_carries_signal(frame):
            return frame
        missing = [
            column
            for column in US_CHILDCARE_REQUIRED_SOURCE_COLUMNS
            if column not in person
        ]
        raise ValueError(
            "US childcare stage cannot heal a default surface without measured "
            f"ASEC source column(s): {missing}."
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_role_metadata(person, entity="person"):
        predictors = _person_childcare_predictors(frame)
        for column in _PUF_PREDICTORS:
            stage_person[_PUF_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        us_childcare_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_childcare_inputs": derive_us_childcare_from_manifest,
            "impute_childcare_to_puf_support": (
                impute_us_childcare_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned_people = output.set_index("person_id").reindex(person["person_id"])
    if aligned_people[_OUTPUT].isna().any():
        raise ValueError(
            "US childcare stage output does not cover every person before "
            "SPM-unit reduction."
        )
    unit_values = (
        aligned_people.assign(
            person_spm_unit_id=person["person_spm_unit_id"].to_numpy()
        )
        .groupby("person_spm_unit_id", sort=False)[_OUTPUT]
        .first()
    )
    aligned_units = unit_values.reindex(spm_unit["spm_unit_id"])
    if aligned_units.isna().any():
        raise ValueError("US childcare stage output does not cover every SPM unit.")

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["spm_unit"][_OUTPUT] = aligned_units.to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_childcare_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and validity diagnostics for childcare."""

    values = pd.to_numeric(frame.table("spm_unit")[_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    weights = np.asarray(frame.resolve_weights("spm_unit").values, dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    total_weight = float(weights.sum())
    positive_share = (
        float(weights[positive].sum()) / total_weight if total_weight > 0.0 else 0.0
    )
    return {
        "positive_share": positive_share,
        "positive_share_band": list(_NONZERO_SHARE_BAND),
        "weighted_total": float(np.sum(np.nan_to_num(values) * weights)),
        "nonfinite": int(np.count_nonzero(~finite)),
        "negative": int(np.count_nonzero(finite & (values < 0.0))),
    }


def us_childcare_signal_gate(frame: Frame) -> GateResult:
    """Require finite, nonnegative, nondefault SPM-unit childcare signal."""

    spm_unit = frame.table("spm_unit")
    if _OUTPUT not in spm_unit:
        return GateResult(
            name="childcare_inputs_signal",
            passed=False,
            failures=(f"spm_unit columns missing: [{_OUTPUT!r}].",),
            details={"missing": [_OUTPUT]},
        )

    summary = us_childcare_summary(frame)
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
    return GateResult(
        name="childcare_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _childcare_surface_carries_signal(frame: Frame) -> bool:
    if _OUTPUT not in frame.table("spm_unit"):
        return False
    return us_childcare_signal_gate(frame).passed
