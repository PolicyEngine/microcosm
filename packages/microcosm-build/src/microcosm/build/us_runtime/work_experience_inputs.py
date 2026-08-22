"""Work-experience industry recodes and the worked-last-year indicator.

The retired eCPS build carried the occupation of the longest job
(``POCCU2`` -> ``detailed_occupation_recode``) but never surfaced its
industry siblings or an explicit worked-last-year indicator.  This stage is
net-new factual-input coverage with no archived derivation to port: the
official ASEC work-experience recodes ``WEIND`` (industry of longest job by
detailed groups, 0--23) and ``WEMIND`` (industry of longest job by major
industry groups, 0--15) carry directly, and ``worked_last_year`` derives as
``WKSWORK > 0`` — the official universe condition of the work-experience
longest-job recode block, verified exact against every pinned archive
(``WEIND`` holds a worker code 1--22 iff ``WKSWORK > 0``; ``WORKYN = 1``
under-covers that universe by ~600 allocation rows per year and is
deliberately not the indicator).

The frozen census_cps inputs never carried ``WEIND``/``WEMIND``, so the
pinned work-experience sidecar (:mod:`.work_experience_source`) restores
them for every pooled income year via exact ``PERIDNUM`` joins before the
derivation runs; ``WKSWORK`` is a frozen census_cps person column.  Both
support clones of one source person inherit identical values through the
shared source identity, matching how ``detailed_occupation_recode`` treats
the PUF tax-detail half.  Industry-conditional modeling stays owned by
PolicyEngine-US; this stage persists measured facts only.
"""

from __future__ import annotations

from importlib.resources import files

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
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_role_series,
)
from microcosm.build.us_runtime.work_experience_source import (
    fill_asec_work_experience_source,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_WORK_EXPERIENCE_NONCONSTANT_PERSON_COLUMNS",
    "US_WORK_EXPERIENCE_OUTPUT_COLUMNS",
    "US_WORK_EXPERIENCE_REQUIRED_SOURCE_COLUMNS",
    "US_WORK_EXPERIENCE_STAGE_NAME",
    "derive_us_work_experience_inputs_from_manifest",
    "us_work_experience_signal_gate",
    "us_work_experience_stage_spec",
    "us_work_experience_summary",
    "with_us_work_experience_inputs",
]

US_WORK_EXPERIENCE_STAGE_NAME = "work_experience_inputs"

_DETAILED_OUTPUT = "detailed_industry_recode"
_MAJOR_OUTPUT = "major_industry_recode"
_WORKED_OUTPUT = "worked_last_year"

US_WORK_EXPERIENCE_OUTPUT_COLUMNS: tuple[str, ...] = (
    _DETAILED_OUTPUT,
    _MAJOR_OUTPUT,
    _WORKED_OUTPUT,
)
US_WORK_EXPERIENCE_NONCONSTANT_PERSON_COLUMNS = US_WORK_EXPERIENCE_OUTPUT_COLUMNS

_DETAILED_SOURCE = "WEIND"
_MAJOR_SOURCE = "WEMIND"
_WEEKS_SOURCE = "WKSWORK"

US_WORK_EXPERIENCE_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    _DETAILED_SOURCE,
    _MAJOR_SOURCE,
    _WEEKS_SOURCE,
)

_PERSON_WEIGHT_COLUMN = "person_weight"
_DETAILED_MAX = 23
_DETAILED_WORKER_MAX = 22
_DETAILED_MILITARY_CODE = 22
_DETAILED_NEVER_WORKED_CODE = 23
_MAJOR_MAX = 15

# Deliberately broad pool-level plausibility bands. The pinned official
# archives measure A_FNLWGT-weighted worked (WKSWORK > 0) shares of
# 0.5200/0.5232/0.5228 and nonzero-recode shares of 0.8217/0.8220/0.8244
# across income years 2022-2024; the pool reweights and clones that
# population without changing either concept, so these floors reject a
# defaulted or collapsed surface while allowing support selection and
# adjacent ASEC vintages ample room to move.
_WORKED_SHARE_BAND = (0.40, 0.62)
_RECODE_SHARE_BAND = (0.70, 0.92)
_DERIVE_WORK_EXPERIENCE_PARAMETER_KEYS = frozenset()


def us_work_experience_stage_spec() -> SourceStageSpec:
    """Load the packaged ``work_experience_inputs`` source-stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_WORK_EXPERIENCE_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_WORK_EXPERIENCE_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_WORK_EXPERIENCE_STAGE_NAME]
    missing = sorted(set(US_WORK_EXPERIENCE_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_WORK_EXPERIENCE_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _bounded_integer_source(
    frame: pd.DataFrame, column: str, *, upper: int
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & (values == np.floor(values))
    valid &= (values >= 0.0) & (values <= float(upper))
    if not valid.all():
        rows = np.flatnonzero(~valid)[:5].tolist()
        raise SourceRuntimeError(
            f"US work-experience source {column!r} must be an integer in "
            f"[0, {upper}] at row(s): {rows}."
        )
    return values.astype(np.int64)


def derive_us_work_experience_inputs_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    _context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Carry the industry recodes and derive the worked-last-year indicator."""

    if operation.kind != "derive_work_experience_inputs":
        raise SourceRuntimeError(
            "US work-experience derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US work-experience derivation requires the person table to be read first."
        )
    unexpected = sorted(
        set(operation.parameters) - _DERIVE_WORK_EXPERIENCE_PARAMETER_KEYS
    )
    if unexpected:
        raise SourceRuntimeError(
            "US work-experience derivation received unsupported parameter(s): "
            f"{unexpected}."
        )
    missing = [
        column
        for column in US_WORK_EXPERIENCE_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US work-experience derivation requires source column(s): {missing}."
        )

    detailed = _bounded_integer_source(frame, _DETAILED_SOURCE, upper=_DETAILED_MAX)
    major = _bounded_integer_source(frame, _MAJOR_SOURCE, upper=_MAJOR_MAX)
    weeks = _bounded_integer_source(frame, _WEEKS_SOURCE, upper=52)
    worked = weeks > 0
    worker_code = (detailed >= 1) & (detailed <= _DETAILED_WORKER_MAX)
    universe_breaks = int(np.count_nonzero(worker_code != worked))
    if universe_breaks:
        raise SourceRuntimeError(
            "US work-experience derivation breaks the recode universe "
            f"identity (WEIND in 1..{_DETAILED_WORKER_MAX} iff WKSWORK > 0) "
            f"on {universe_breaks} row(s)."
        )
    recode_zero_breaks = int(np.count_nonzero((detailed == 0) != (major == 0)))
    if recode_zero_breaks:
        raise SourceRuntimeError(
            "US work-experience derivation finds the detailed and major "
            f"recodes disagreeing on not-in-universe rows: "
            f"{recode_zero_breaks} row(s)."
        )
    result = frame.copy(deep=True)
    result[_DETAILED_OUTPUT] = detailed.astype(np.int16)
    result[_MAJOR_OUTPUT] = major.astype(np.int16)
    result[_WORKED_OUTPUT] = worked
    return result


def with_us_work_experience_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    asec_work_experience_source: pd.DataFrame | None = None,
) -> Frame:
    """Materialize work-experience inputs on a US frame.

    The frozen census_cps inputs never carried raw ASEC ``WEIND`` or
    ``WEMIND``, so when the person table lacks them the pinned
    work-experience sidecar (:mod:`.work_experience_source`) must be
    supplied; the fill is an exact per-income-year ``PERIDNUM`` join and
    never predicts a value.  ``WKSWORK`` is a frozen census_cps person
    column and must already be present.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US work-experience inputs require the US schema.")
    if _work_experience_surface_carries_signal(frame):
        return frame

    person = frame.table("person")
    stage_person = person.copy(deep=True)
    if _WEEKS_SOURCE not in stage_person.columns:
        raise SourceRuntimeError(
            "US work-experience stage requires the frozen census_cps person "
            f"column {_WEEKS_SOURCE!r}."
        )
    if any(
        column not in stage_person.columns
        for column in (_DETAILED_SOURCE, _MAJOR_SOURCE)
    ):
        if asec_work_experience_source is None:
            raise SourceRuntimeError(
                "US work-experience stage requires the pinned ASEC "
                "work-experience sidecar to restore WEIND/WEMIND (the frozen "
                "census_cps inputs never carried them); pass "
                "--asec-work-experience-source or allow the official fetch."
            )
        stage_person = fill_asec_work_experience_source(
            stage_person, asec_work_experience_source
        )
    stage_person[_PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    output = run_source_stage(
        us_work_experience_stage_spec(),
        tables={"person": stage_person},
        operation_handlers={
            "derive_work_experience_inputs": (
                derive_us_work_experience_inputs_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(time_period)),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    for column in US_WORK_EXPERIENCE_OUTPUT_COLUMNS:
        if aligned[column].isna().any():
            raise ValueError(
                "US work-experience stage output does not cover every person "
                f"for {column!r}."
            )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][_DETAILED_OUTPUT] = aligned[_DETAILED_OUTPUT].to_numpy(
        dtype=np.int16
    )
    tables["person"][_MAJOR_OUTPUT] = aligned[_MAJOR_OUTPUT].to_numpy(dtype=np.int16)
    tables["person"][_WORKED_OUTPUT] = aligned[_WORKED_OUTPUT].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_work_experience_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and coherence diagnostics for the stage."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    detailed = pd.to_numeric(person[_DETAILED_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    major = pd.to_numeric(person[_MAJOR_OUTPUT], errors="coerce").to_numpy(
        dtype=np.float64
    )
    worked = person[_WORKED_OUTPUT].fillna(False).astype(bool).to_numpy()
    worker_code = (detailed >= 1.0) & (detailed <= float(_DETAILED_WORKER_MAX))
    recode_positive = detailed != 0.0

    def _share(mask: np.ndarray) -> float:
        return float(weights[mask].sum()) / total_weight if total_weight > 0 else 0.0

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
                "worked_share": (
                    float(weights[mask & worked].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
                "recode_positive_share": (
                    float(weights[mask & recode_positive].sum()) / channel_weight
                    if channel_weight > 0.0
                    else 0.0
                ),
            }
    return {
        "worked_share": _share(worked),
        "recode_positive_share": _share(recode_positive),
        "military_share": _share(detailed == float(_DETAILED_MILITARY_CODE)),
        "never_worked_share": _share(detailed == float(_DETAILED_NEVER_WORKED_CODE)),
        "channels": channels,
        "worked_share_band": list(_WORKED_SHARE_BAND),
        "recode_share_band": list(_RECODE_SHARE_BAND),
        "nonfinite_detailed": int(np.count_nonzero(~np.isfinite(detailed))),
        "nonfinite_major": int(np.count_nonzero(~np.isfinite(major))),
        "out_of_range_detailed": int(
            np.count_nonzero((detailed < 0.0) | (detailed > float(_DETAILED_MAX)))
        ),
        "out_of_range_major": int(
            np.count_nonzero((major < 0.0) | (major > float(_MAJOR_MAX)))
        ),
        "universe_identity_breaks": int(np.count_nonzero(worker_code != worked)),
        "recode_zero_breaks": int(
            np.count_nonzero((detailed == 0.0) != (major == 0.0))
        ),
    }


def us_work_experience_signal_gate(frame: Frame) -> GateResult:
    """Require nonzero, plausible, coherent work-experience signal."""

    person = frame.table("person")
    missing = [
        column
        for column in US_WORK_EXPERIENCE_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    if missing:
        return GateResult(
            name="work_experience_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_work_experience_summary(frame)
    failures: list[str] = []
    for count_key, label in (
        ("nonfinite_detailed", "detailed_industry_recode nonfinite values"),
        ("nonfinite_major", "major_industry_recode nonfinite values"),
        ("out_of_range_detailed", "detailed_industry_recode out-of-range values"),
        ("out_of_range_major", "major_industry_recode out-of-range values"),
        ("universe_identity_breaks", "worker-code rows disagreeing with worked"),
        ("recode_zero_breaks", "detailed/major zero-row disagreements"),
    ):
        count = int(summary[count_key])
        if count:
            failures.append(f"{label}: {count}.")

    for share_key, band_key, label in (
        ("worked_share", "worked_share_band", "worked-last-year share"),
        ("recode_positive_share", "recode_share_band", "industry-recode share"),
    ):
        share = float(summary[share_key])
        low, high = summary[band_key]
        if not (low <= share <= high):
            failures.append(
                f"{label} {share:.3f} outside plausibility band [{low}, {high}]."
            )

    return GateResult(
        name="work_experience_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )


def _work_experience_surface_carries_signal(frame: Frame) -> bool:
    person = frame.table("person")
    if not all(column in person for column in US_WORK_EXPERIENCE_OUTPUT_COLUMNS):
        return False
    return us_work_experience_signal_gate(frame).passed
