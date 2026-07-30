"""Education-credit inputs from IRS PUF tuition and CPS ASEC assistance.

The retired eCPS build derived person qualified tuition as the larger of IRS
PUF fields E03230 and E87530 (falling back to E03230) and carried educational
assistance directly from CPS ASEC ``ED_VAL``.  In the archived publication
path at ``9f8db56e9ed1ca25f0ed16285fcc928343d2e474``,
``calibration/puf_impute.py`` drops the reported PUF
``american_opportunity_credit`` output before
``datasets/cps/extended_cps.py::_impute_aotc_eligibility_inputs`` runs.  That
method therefore takes its documented fallback: every positive-tuition person
gets the five affirmative American Opportunity Tax Credit factual inputs.  The
pinned reference independently confirms the published result: qualified
tuition and all five flags have exactly the same support mask.

This stage is intentionally factual-input-only.  It does not persist computed
education credits or eligibility formulas owned by PolicyEngine-US.
"""

from __future__ import annotations

from importlib.resources import files

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
from populace.build.us_runtime.education_assistance_source import (
    fill_asec_education_assistance_source,
)
from populace.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_role_series,
)
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS",
    "US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS",
    "US_EDUCATION_INPUTS_OUTPUT_COLUMNS",
    "US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS",
    "US_EDUCATION_INPUTS_STAGE_NAME",
    "derive_us_education_inputs_from_manifest",
    "us_education_inputs_signal_gate",
    "us_education_inputs_stage_spec",
    "us_education_inputs_summary",
    "with_us_education_inputs",
]

US_EDUCATION_INPUTS_STAGE_NAME = "education_inputs"

US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS: tuple[str, ...] = (
    "is_pursuing_credential_for_american_opportunity_credit",
    "attends_eligible_educational_institution_for_american_opportunity_credit",
    "is_enrolled_at_least_half_time_for_american_opportunity_credit",
    "has_american_opportunity_credit_1098_t_or_exception",
    "has_american_opportunity_credit_institution_ein",
)

US_EDUCATION_INPUTS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "qualified_tuition_expenses",
    "educational_assistance",
    *US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS,
)

US_EDUCATION_INPUTS_NONCONSTANT_PERSON_COLUMNS = US_EDUCATION_INPUTS_OUTPUT_COLUMNS

US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "ED_VAL",
    "qualified_tuition_expenses",
)

_PERSON_WEIGHT_COLUMN = "person_weight"
# Qualified tuition is a PUF-only tax-detail chain target: on the two-channel
# support pool it is exactly zero on the ASEC half BY DESIGN (like every other
# PUF-only detail column — partnership income, charitable donations, mortgage
# interest all decompose the same way on the full-scale pool), and the PUF
# half carries the PUF-faithful per-person rate (~0.56% weighted on the first
# full-scale base, base-r3 ck016). The retired extended-CPS 2.64% person rate
# is a blended post-imputation file rate and is not comparable to this
# pre-selection pool, which is how the original [0.003, 0.05] total floor was
# mis-set. The gate now asserts the channel invariant directly: ASEC exactly
# zero, PUF share within band, blended total within band.
_TUITION_SHARE_BAND = (0.001, 0.05)
_TUITION_PUF_CHANNEL_SHARE_BAND = (0.002, 0.05)
_ASSISTANCE_SHARE_BAND = (0.005, 0.08)
_DERIVE_EDUCATION_INPUTS_PARAMETER_KEYS = frozenset()


def us_education_inputs_stage_spec() -> SourceStageSpec:
    """Load the packaged ``education_inputs`` source-stage declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_EDUCATION_INPUTS_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_EDUCATION_INPUTS_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_EDUCATION_INPUTS_STAGE_NAME]
    missing = sorted(set(US_EDUCATION_INPUTS_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_EDUCATION_INPUTS_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def derive_us_education_inputs_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Derive the seven education inputs from a person source table."""

    if operation.kind != "derive_education_inputs":
        raise SourceRuntimeError(
            "US education-input derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US education-input derivation requires the person table to be read first."
        )
    unexpected = sorted(
        set(operation.parameters) - _DERIVE_EDUCATION_INPUTS_PARAMETER_KEYS
    )
    if unexpected:
        raise SourceRuntimeError(
            "US education-input derivation received unsupported parameter(s): "
            f"{unexpected}."
        )
    missing = [
        column
        for column in US_EDUCATION_INPUTS_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US education-input derivation requires source column(s): {missing}."
        )

    result = frame.copy(deep=True)
    tuition = (
        pd.to_numeric(result["qualified_tuition_expenses"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    assistance = (
        pd.to_numeric(result["ED_VAL"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
    aotc_student = tuition > 0.0
    result["qualified_tuition_expenses"] = tuition.to_numpy(dtype=np.float64)
    result["educational_assistance"] = assistance.to_numpy(dtype=np.float64)
    for column in US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS:
        result[column] = aotc_student.to_numpy(dtype=bool)
    return result


def with_us_education_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    asec_education_source: pd.DataFrame | None = None,
) -> Frame:
    """Materialize education inputs on a US frame, healing default surfaces.

    The frozen census_cps inputs never carried raw ASEC ``ED_VAL``, so when
    the person table lacks it the pinned education-assistance sidecar
    (:mod:`.education_assistance_source`) must be supplied; the fill is an
    exact per-income-year ``PERIDNUM`` join and never predicts a value.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US education inputs require the US schema.")
    if _education_surface_carries_signal(frame):
        return frame

    person = frame.table("person")
    stage_person = person.copy(deep=True)
    if "ED_VAL" not in stage_person.columns:
        if asec_education_source is None:
            raise SourceRuntimeError(
                "US education-inputs stage requires the pinned ASEC "
                "education-assistance sidecar to restore ED_VAL (the frozen "
                "census_cps inputs never carried it); pass "
                "--asec-education-source or allow the official fetch."
            )
        stage_person = fill_asec_education_assistance_source(
            stage_person, asec_education_source
        )
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_education_inputs_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_education_inputs": derive_us_education_inputs_from_manifest,
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_EDUCATION_INPUTS_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                "US education-input stage output does not cover every person for "
                f"{column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column in US_EDUCATION_INPUTS_OUTPUT_COLUMNS:
        if column in US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS:
            tables["person"][column] = aligned[column].to_numpy(dtype=bool)
        else:
            tables["person"][column] = aligned[column].to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_education_inputs_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and coherence diagnostics for education inputs."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    tuition = pd.to_numeric(
        person["qualified_tuition_expenses"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    assistance = pd.to_numeric(
        person["educational_assistance"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    tuition_positive = np.isfinite(tuition) & (tuition > 0.0)
    assistance_positive = np.isfinite(assistance) & (assistance > 0.0)

    def _share(mask: np.ndarray) -> float:
        return float(weights[mask].sum()) / total_weight if total_weight > 0 else 0.0

    incoherent: dict[str, int] = {}
    for column in US_AOTC_ELIGIBILITY_OUTPUT_COLUMNS:
        values = person[column].fillna(False).astype(bool).to_numpy()
        incoherent[column] = int(np.count_nonzero(values != tuition_positive))
    channels: dict[str, dict[str, float | int]] = {}
    if has_support_role_metadata(person, entity="person"):
        channel = support_role_series(person, entity="person").to_numpy()
        for name in (
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ):
            mask = channel == name
            channel_weight = float(weights[mask].sum())
            channels[name] = {
                "rows": int(np.count_nonzero(mask)),
                "tuition_positive_rows": int(np.count_nonzero(mask & tuition_positive)),
                "tuition_positive_share": (
                    float(weights[mask & tuition_positive].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
                "assistance_positive_share": (
                    float(weights[mask & assistance_positive].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
            }
    return {
        "qualified_tuition_share": _share(tuition_positive),
        "educational_assistance_share": _share(assistance_positive),
        "channels": channels,
        "qualified_tuition_total": float(np.nansum(tuition)),
        "educational_assistance_total": float(np.nansum(assistance)),
        "tuition_share_band": list(_TUITION_SHARE_BAND),
        "assistance_share_band": list(_ASSISTANCE_SHARE_BAND),
        "nonfinite_tuition": int(np.count_nonzero(~np.isfinite(tuition))),
        "nonfinite_assistance": int(np.count_nonzero(~np.isfinite(assistance))),
        "negative_tuition": int(np.count_nonzero(tuition < 0.0)),
        "negative_assistance": int(np.count_nonzero(assistance < 0.0)),
        "incoherent_aotc_flags": incoherent,
    }


def us_education_inputs_signal_gate(frame: Frame) -> GateResult:
    """Require nonzero, plausible, coherent education-input signal."""

    person = frame.table("person")
    missing = [
        column
        for column in US_EDUCATION_INPUTS_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="education_inputs_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_education_inputs_summary(frame)
    failures: list[str] = []
    for count_key, label in (
        ("nonfinite_tuition", "qualified_tuition_expenses nonfinite values"),
        ("nonfinite_assistance", "educational_assistance nonfinite values"),
        ("negative_tuition", "qualified_tuition_expenses negative values"),
        ("negative_assistance", "educational_assistance negative values"),
    ):
        count = int(summary[count_key])
        if count:
            failures.append(f"{label}: {count}.")

    for share_key, band_key, label in (
        (
            "qualified_tuition_share",
            "tuition_share_band",
            "qualified-tuition share",
        ),
        (
            "educational_assistance_share",
            "assistance_share_band",
            "educational-assistance share",
        ),
    ):
        share = float(summary[share_key])
        low, high = summary[band_key]
        if not (low <= share <= high):
            failures.append(
                f"{label} {share:.3f} outside plausibility band [{low}, {high}]."
            )

    for column, count in summary["incoherent_aotc_flags"].items():
        if count:
            failures.append(
                f"{column}: {count} rows disagree with positive qualified tuition."
            )

    channels = summary.get("channels") or {}
    if channels:
        asec = channels.get(BASE_ASEC_SUPPORT_CHANNEL, {})
        if int(asec.get("tuition_positive_rows", 0)):
            failures.append(
                "qualified_tuition_expenses is a PUF-only tax-detail column but "
                f"{asec['tuition_positive_rows']} ASEC-channel row(s) carry it; "
                "the support channels have cross-contaminated."
            )
        puf = channels.get(PUF_TAX_DETAIL_SUPPORT_CHANNEL, {})
        puf_share = float(puf.get("tuition_positive_share", 0.0))
        puf_low, puf_high = _TUITION_PUF_CHANNEL_SHARE_BAND
        if not (puf_low <= puf_share <= puf_high):
            failures.append(
                f"puf_tax_detail qualified-tuition share {puf_share:.4f} outside "
                f"plausibility band [{puf_low}, {puf_high}]."
            )
    return GateResult(
        name="education_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _education_surface_carries_signal(frame: Frame) -> bool:
    person = frame.table("person")
    if not all(column in person for column in US_EDUCATION_INPUTS_OUTPUT_COLUMNS):
        return False
    return us_education_inputs_signal_gate(frame).passed
