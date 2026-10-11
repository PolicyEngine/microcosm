"""Joint SCF participation/amount allocation of CPS retirement contributions.

A complete observed donor vector supplies IRA-only, employee-plan-only, or
both participation. CPS RETCB_VAL scales the whole vector, while a second
whole-vector hot deck replaces PUF support values. Statutory contributions,
active participation and deductions remain model-derived. SCF source-period
and account-type proxy limitations are declared in the source manifest.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

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
from microcosm.build.us_runtime.joint_retirement import (
    RETIREMENT_COLUMNS,
    draw_joint_retirement_vectors,
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

US_RETIREMENT_CONTRIBUTION_STAGE_NAME = "retirement_contributions"

US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS: tuple[str, ...] = RETIREMENT_COLUMNS
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
    "employment_income",
    "self_employment_income",
)
_PUF_PREDICTOR_PREFIX = "retirement_contribution_predictor_"
_JOINT_DERIVATION_PARAMETER_KEYS = frozenset(
    {"donor_table", "age_breaks", "seed_from_build_config", "roth_dc_share"}
)
_PUF_IMPUTATION_PARAMETER_KEYS = frozenset(
    {"predictors", "age_breaks", "seed_from_build_config", "weight"}
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


def _joint_parameters(operation: SourceOperationSpec, keys: frozenset[str]) -> None:
    unexpected = sorted(set(operation.parameters) - keys)
    missing = sorted(keys - set(operation.parameters))
    if unexpected or missing:
        raise SourceRuntimeError(
            f"Joint retirement parameters: missing={missing}, unexpected={unexpected}."
        )
    if operation.parameters["seed_from_build_config"] is not True:
        raise SourceRuntimeError("Joint retirement seed must come from build config.")


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
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Scale a whole empirical SCF vector to each positive CPS bundled total."""
    if operation.kind != "derive_retirement_contributions" or frame is None:
        raise SourceRuntimeError(
            "Joint retirement derivation requires a person table and derive operation."
        )
    _joint_parameters(operation, _JOINT_DERIVATION_PARAMETER_KEYS)
    missing = sorted(
        set(US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS) - set(frame)
    )
    if missing:
        raise SourceRuntimeError(
            f"Joint retirement derivation requires measured ASEC source columns: {missing}."
        )
    if context is None:
        raise SourceRuntimeError(
            "Joint retirement derivation requires the declared SCF donor table."
        )
    donor = context.read_table(str(operation.parameters["donor_table"]))
    result = frame.copy(deep=True)
    source = _numeric_source(result, "RETCB_VAL")
    if (source < 0).any():
        raise SourceRuntimeError("US retirement RETCB_VAL contains negative values.")
    wages = _numeric_source(result, "WSAL_VAL")
    se = _numeric_source(result, "SEMP_VAL")
    age_column = "age" if "age" in result else "A_AGE"
    if age_column not in result:
        raise SourceRuntimeError(
            "Joint retirement derivation requires measured age or A_AGE."
        )
    recipient = pd.DataFrame(
        {
            "person_id": result["person_id"].to_numpy(),
            "age": _numeric_source(result, age_column),
            "has_wages": wages > 0,
            "has_self_employment": se > 0,
            "rank_value": source,
            "person_weight": _numeric_source(result, _PERSON_WEIGHT_COLUMN),
        },
        index=result.index,
    )
    donor["rank_value"] = donor[list(RETIREMENT_COLUMNS)].sum(axis=1)
    positive = source > 0
    values = pd.DataFrame(0.0, index=result.index, columns=RETIREMENT_COLUMNS)
    if positive.any():
        draws = draw_joint_retirement_vectors(
            donor,
            recipient.loc[positive],
            donor_weight="scf_weight",
            recipient_weight=_PERSON_WEIGHT_COLUMN,
            seed=context.config.seed,
            age_breaks=operation.parameters["age_breaks"],
            positive_only=True,
        )
        donor_total = draws.sum(axis=1).to_numpy()
        values.loc[positive] = (
            draws.to_numpy() * (source[positive] / donor_total)[:, None]
        )
        result.attrs["retirement_cps_draw"] = draws.attrs["joint_retirement"]
    result[list(RETIREMENT_COLUMNS)] = values
    return result


def impute_us_retirement_contributions_to_puf_support_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Draw a complete ASEC vector for each PUF row, including nonsavers."""
    if (
        operation.kind != "impute_retirement_contributions_to_puf_support"
        or frame is None
    ):
        raise SourceRuntimeError(
            "Joint retirement PUF imputation requires person table and expected operation."
        )
    _joint_parameters(operation, _PUF_IMPUTATION_PARAMETER_KEYS)
    if not has_support_role_metadata(frame, entity="person"):
        return frame.copy(deep=True)
    if context is None:
        raise SourceRuntimeError(
            "Joint retirement PUF imputation requires build context."
        )
    predictors = tuple(operation.parameters["predictors"])
    if (
        predictors != _PUF_PREDICTORS
        or operation.parameters["weight"] != _PERSON_WEIGHT_COLUMN
    ):
        raise SourceRuntimeError(
            "Joint retirement PUF predictors or person-weight column drifted."
        )
    required = [
        _PERSON_WEIGHT_COLUMN,
        *(_PUF_PREDICTOR_PREFIX + p for p in predictors),
        *RETIREMENT_COLUMNS,
    ]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise SourceRuntimeError(
            f"Joint retirement PUF imputation missing columns: {missing}."
        )
    roles = support_role_series(frame, entity="person")
    asec, puf = (
        roles.eq(_BASE_ASEC_SUPPORT_CHANNEL),
        roles.eq(_PUF_TAX_DETAIL_SUPPORT_CHANNEL),
    )
    if not asec.any() or not puf.any():
        raise SourceRuntimeError(
            "Joint retirement PUF imputation requires nonempty ASEC and PUF support roles."
        )

    def inputs(mask: pd.Series, identity: str) -> pd.DataFrame:
        table = frame.loc[mask].copy()
        table[identity] = table["person_id"]
        table["age"] = _numeric_source(table, _PUF_PREDICTOR_PREFIX + "age")
        wage = _numeric_source(table, _PUF_PREDICTOR_PREFIX + "employment_income")
        se = _numeric_source(table, _PUF_PREDICTOR_PREFIX + "self_employment_income")
        table["has_wages"], table["has_self_employment"] = wage > 0, se > 0
        table["rank_value"] = np.maximum(wage, 0) + np.maximum(se, 0)
        return table

    donor = inputs(asec, "donor_id")
    recipient = inputs(puf, "person_id")
    predictions = draw_joint_retirement_vectors(
        donor,
        recipient,
        donor_weight=_PERSON_WEIGHT_COLUMN,
        recipient_weight=_PERSON_WEIGHT_COLUMN,
        seed=context.config.seed,
        age_breaks=operation.parameters["age_breaks"],
        positive_only=False,
    )
    result = frame.copy(deep=True)
    result.loc[puf, list(RETIREMENT_COLUMNS)] = predictions.to_numpy()
    result.attrs["retirement_puf_draw"] = predictions.attrs["joint_retirement"]
    return result


def _person_retirement_predictors(frame: Frame) -> pd.DataFrame:
    """Construct the declared age and earned-income hot-deck predictors."""
    person = frame.table("person")
    predictors = pd.DataFrame(index=person.index)
    for target, sources in {
        "age": ("age", "A_AGE"),
        "employment_income": ("employment_income_before_lsr", "WSAL_VAL"),
        "self_employment_income": ("self_employment_income_before_lsr", "SEMP_VAL"),
    }.items():
        column = next((column for column in sources if column in person), None)
        if column is None:
            raise SourceRuntimeError(
                f"Joint retirement predictor {target!r} requires one of {sources}."
            )
        predictors[target] = _numeric_source(person, column)
    return predictors


def with_us_retirement_contribution_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    donor: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
) -> Frame:
    """Materialize source contributions, rebuilding legacy fixed-share leaves.

    Donor injection is for explicit source tables and hermetic contract tests.
    Ordinary builds fetch the byte-pinned public SCF into their local cache.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US retirement contributions require the US schema.")
    person = frame.table("person")
    has_support_roles = has_support_role_metadata(person, entity="person")
    spec = us_retirement_contributions_stage_spec()
    if donor is None:
        from microcosm.build.us_runtime.scf_retirement import load_scf_retirement_donor

        donor = load_scf_retirement_donor(
            cache_dir=cache_dir
            or Path.home() / ".cache" / "microcosm" / "scf_retirement",
            roth_dc_share=float(spec.operations[1].parameters["roth_dc_share"]),
        )

    stage_person = person.copy(deep=True)
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    if has_support_roles:
        predictors = _person_retirement_predictors(frame)
        for column in _PUF_PREDICTORS:
            stage_person[_PUF_PREDICTOR_PREFIX + column] = predictors[column].to_numpy()
    output = run_source_stage(
        spec,
        tables={"person": stage_person, "scf_retirement": donor},
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
        metadata={
            **frame.metadata,
            "retirement_contributions": {
                "method": "weighted_joint_hot_deck_v1",
                "seed": int(seed),
                "donor": dict(donor.attrs.get("scf_retirement", {})),
                "cps_draw": output.attrs.get("retirement_cps_draw"),
                "puf_draw": output.attrs.get("retirement_puf_draw"),
            },
        },
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
    allocation_mismatch = reconciliation_mask & ~np.isclose(
        combined,
        source,
        rtol=1e-9,
        atol=1e-6,
    )
    ira = (contributions["traditional_ira_contributions_desired"] > 0) | (
        contributions["roth_ira_contributions_desired"] > 0
    )
    traditional = contributions["traditional_ira_contributions_desired"] > 0
    employee_plan = (
        (contributions["traditional_401k_contributions_desired"] > 0)
        | (contributions["roth_401k_contributions_desired"] > 0)
        | (contributions["self_employed_pension_contributions_desired"] > 0)
    )
    participation = {
        name: {
            "records": int(mask.sum()),
            "weighted_people": float(weights[mask].sum()),
        }
        for name, mask in {
            "ira_only": ira & ~employee_plan,
            "employer_only": ~ira & employee_plan,
            "both": ira & employee_plan,
            "neither": ~ira & ~employee_plan,
            "traditional_ira_contributors": traditional,
        }.items()
    }
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
        "joint_participation": participation,
        "traditional_ira_contributor_employee_plan_share": (
            float(
                weights[traditional & employee_plan].sum() / weights[traditional].sum()
            )
            if weights[traditional].sum() > 0
            else 0.0
        ),
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
    for cell in ("ira_only", "employer_only", "both"):
        if summary["joint_participation"][cell]["weighted_people"] <= 0:
            failures.append(
                f"Joint retirement participation has no positive-weight {cell} rows."
            )
    if summary["allocation_mismatch_rows"]:
        failures.append(
            f"ASEC bundled totals do not reconcile on {summary['allocation_mismatch_rows']} direct-source rows."
        )
    return GateResult(
        name="retirement_contributions_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
