"""Usual-hours production and coverage for the retained ACS local lane.

This is a fresh-build repair for #765, not a patcher for a published H5. A
stored value of 0 or 40 is never itself evidence of missingness. A historical
artifact whose consumer defaults replaced nulls needs its original lineage
before those cells can be repaired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import (
    ASEC_PUF_DONOR_SPINE,
    TargetFamilies,
    declared_acs_transfer_target_families,
    resolve_acs_donor_channel,
)
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.build.us_runtime.hours_worked import (
    US_HOURS_WORKED_OUTPUT_COLUMNS,
    US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
    US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS,
    with_us_hours_worked_inputs,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    has_assembled_support_metadata,
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import Frame

_USUAL_HOURS = US_HOURS_WORKED_POOL_OUTPUT_COLUMNS[0]
ACS_UNDER15_ZERO_POLICY = "us_hours_under15_zero_completion_v1"
# Census 2025 ASEC dictionary, person-record labor-force fields, page 6C-20:
# https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf
# HRSWK 99 means 99+ hours. A_HRS1 -1 is NIU, mapped to zero only by the
# existing source producer. Other sentinels and out-of-domain values fail.
_RAW_HOURS_RANGES = {
    "HRSWK": (0, 99),
    "A_HRS1": (-1, 99),
    "WKSWORK": (0, 52),
    "WORKYN": (1, 2),
    "WTEMP": (0, 2),
    "WRK_CK": (1, 2),
}


def acs_local_transfer_target_families() -> TargetFamilies:
    """Add the qualified hours leaf only to the local builder's explicit plan."""
    plan = {
        entity: dict(families)
        for entity, families in declared_acs_transfer_target_families().items()
    }
    plan["person"]["source_operator_hours_worked"] = (_USUAL_HOURS,)
    return plan


def acs_local_hours_transfer_target_families() -> TargetFamilies:
    """The separate ASEC-only transfer pass, excluding PUF tax-detail targets."""
    return {"person": {"source_operator_hours_worked": (_USUAL_HOURS,)}}


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def require_acs_local_hours_fallback_universe(frame: Frame) -> None:
    """Refuse modeling survey-universe absence as an observed zero.

    ASEC labor questions cover age 15+, so they can supply an age-15
    recipient missing ACS WKHP. Younger or unknown-age recipients need a
    separately reviewed modeled-completion policy before this transfer.
    """
    person = frame.person
    missing = (
        person[_USUAL_HOURS].isna()
        if _USUAL_HOURS in person
        else pd.Series(True, index=person.index)
    )
    age = pd.to_numeric(
        person.get("age", pd.Series(np.nan, index=person.index)), errors="coerce"
    )
    eligible = np.isfinite(age) & age.ge(15) & age.eq(np.floor(age))
    if (missing & ~eligible).any():
        raise ValueError(
            "Unresolved ACS usual hours below age 15 or with unknown age require "
            "an explicit modeled-completion policy; survey NIU is not observed zero."
        )


def complete_acs_local_under15_hours(
    frame: Frame, *, policy: str | None = None
) -> tuple[Frame, dict[str, object] | None]:
    """Explicitly model unresolved under-15 ACS hours as zero, if selected.

    This is a tax-benefit input assumption, not evidence that each child
    actually worked zero hours. Native provenance and raw NIU values stay
    unchanged. Missing earnings are counted as unknown, not as no earnings.
    """
    if policy is None:
        return frame, None
    if policy != ACS_UNDER15_ZERO_POLICY:
        raise ValueError(f"Unsupported ACS hours completion policy: {policy!r}.")
    person = frame.person
    tag = spine_column("person")
    if tag in person and not person[tag].eq(ACS_2024_1YR_SPINE).all():
        raise ValueError("Under-15 hours completion requires an ACS-only frame.")
    age = pd.to_numeric(
        person.get("age", pd.Series(np.nan, index=person.index)), errors="coerce"
    )
    known_age = np.isfinite(age) & age.ge(0) & age.eq(np.floor(age))
    missing = (
        person[_USUAL_HOURS].isna()
        if _USUAL_HOURS in person
        else pd.Series(True, index=person.index)
    )
    if (missing & ~known_age).any():
        raise ValueError("Under-15 hours completion cannot resolve unknown age.")
    selected = missing & age.lt(15)
    conflicts = pd.Series(False, index=person.index)
    for column in (
        "WKHP",
        "WAGP",
        "employment_income_before_lsr",
    ):
        if column in person:
            conflicts |= pd.to_numeric(person[column], errors="coerce").gt(0)
    for column in ("SEMP", "self_employment_income_before_lsr"):
        if column in person:
            values = pd.to_numeric(person[column], errors="coerce")
            conflicts |= values.notna() & values.ne(0)
    if "WKL" in person:
        conflicts |= pd.to_numeric(person["WKL"], errors="coerce").eq(1)
    if (selected & conflicts).any():
        raise ValueError(
            "Under-15 hours completion contradicts work or nonzero earnings evidence."
        )
    unknown_earnings = pd.Series(False, index=person.index)
    earnings_columns = []
    earnings_evidence = []
    for candidates in (
        ("WAGP", "employment_income_before_lsr"),
        ("SEMP", "self_employment_income_before_lsr"),
    ):
        column = next(
            (candidate for candidate in candidates if candidate in person), None
        )
        earnings_columns.append(column)
        earnings_evidence.append(
            {
                "column": column,
                "scope": "unavailable"
                if column is None
                else ("raw_source" if column == candidates[0] else "mapped_input"),
            }
        )
        unknown_earnings |= (
            ~np.isfinite(pd.to_numeric(person[column], errors="coerce"))
            if column is not None
            else pd.Series(True, index=person.index)
        )
    result = frame
    if selected.any():
        updated = person.copy(deep=True)
        if _USUAL_HOURS not in updated:
            updated[_USUAL_HOURS] = np.nan
        updated.loc[selected, _USUAL_HOURS] = 0.0
        result = _with_person(frame, updated)
    return result, {
        "policy": policy,
        "version": 1,
        "provenance": "modeled_assumption",
        "entity": "person",
        "column": _USUAL_HOURS,
        "modeled_rows": int(selected.sum()),
        "modeled_rows_by_age": {
            str(int(value)): int(count)
            for value, count in age[selected].value_counts().sort_index().items()
        },
        "earnings_unknown_rows": int((selected & unknown_earnings).sum()),
        "earnings_columns": earnings_columns,
        "earnings_evidence": earnings_evidence,
        "pre_completion_missing_rows": int(missing.sum()),
        "remaining_missing_rows": int((missing & ~selected).sum()),
        "preserved_known_rows": int((~missing).sum()),
    }


def prepare_acs_local_hours_donor(
    frame: Frame, *, seed: int, period: int
) -> tuple[Frame, Frame, dict[str, object]]:
    """Fill absent donor hours from complete raw ASEC fields, preserving inputs.

    The local loader does not execute the fiscal builder's source operators.
    Select the ASEC observation role, then run its existing derivation before
    transfer, retaining only missing usual-hours cells. Source-incomplete
    donors and disagreements with existing observations fail explicitly. This
    does not interpret a PUF-imputed value as a direct ASEC observation, or
    accept an apparently nonconstant column as source qualification.
    """
    original = frame
    original_person = original.table("person")
    if not has_support_role_metadata(original_person, entity="person") or (
        has_assembled_support_metadata(original_person, entity="person")
    ):
        raise ValueError(
            "Local raw hours donor requires explicit legacy ASEC observation roles; "
            "unclassified or multispine records need their own source lineage."
        )
    frame, _ = resolve_acs_donor_channel(frame, BASE_ASEC_SUPPORT_CHANNEL)
    person = frame.table("person")
    if (
        not support_role_series(person, entity="person")
        .eq(BASE_ASEC_SUPPORT_CHANNEL)
        .all()
    ):
        raise ValueError("Local hours donor selection contains non-ASEC roles.")
    if person.empty:
        raise ValueError("Local raw hours donor contains no ASEC observations.")
    source_age_column = "A_AGE" if "A_AGE" in person else "age"
    if source_age_column not in person:
        raise ValueError(
            "Local raw hours donor requires age to establish its universe."
        )
    age = pd.to_numeric(person[source_age_column], errors="coerce")
    in_universe = np.isfinite(age) & age.ge(15) & age.eq(np.floor(age))
    excluded_rows = int((~in_universe).sum())
    if not in_universe.any():
        raise ValueError("Local raw hours donor contains no ASEC age-15+ observations.")
    if not in_universe.all():
        frame = frame.select(in_universe)
        person = frame.person
    work_status_columns = tuple(
        column for column in ("WORKYN", "WTEMP", "WRK_CK") if column in person
    )
    source_columns = (*US_HOURS_WORKED_REQUIRED_SOURCE_COLUMNS, *work_status_columns)
    for column in source_columns:
        if column not in person:
            raise ValueError(f"Local hours donor requires complete raw ASEC {column}.")
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        low, high = _RAW_HOURS_RANGES[column]
        if not (
            np.isfinite(values)
            & (values == np.floor(values))
            & (values >= low)
            & (values <= high)
        ).all():
            raise ValueError(
                f"Local hours donor requires raw ASEC {column} integer codes "
                f"within [{low}, {high}]."
            )
    hours = pd.to_numeric(person["HRSWK"])
    weeks = pd.to_numeric(person["WKSWORK"])
    # At age 15+, the dictionary routes nonworkers outside WKSWORK and
    # HRSWK. Their coherent NIU pair is distinct from children's NIU.
    coherent = (hours.gt(0) & weeks.gt(0)) | (hours.eq(0) & weeks.eq(0))
    # WRK_CK includes temporary/part-time work established by WTEMP after an
    # initial WORKYN=2. An initial no alone does not establish final nonwork.
    if "WRK_CK" in person:
        final_work = pd.to_numeric(person["WRK_CK"])
        coherent &= (final_work.eq(1) & weeks.gt(0)) | (final_work.eq(2) & weeks.eq(0))
    if "WORKYN" in person:
        initial_work = pd.to_numeric(person["WORKYN"])
        coherent &= ~initial_work.eq(1) | weeks.gt(0)
        if "WTEMP" in person:
            temporary_work = pd.to_numeric(person["WTEMP"])
            final_no = initial_work.eq(2) & temporary_work.eq(2)
            coherent &= ~final_no | weeks.eq(0)
    if "WTEMP" in person:
        temporary_work = pd.to_numeric(person["WTEMP"])
        # NIU (0), or an absent follow-up column, is not a negative answer.
        coherent &= ~temporary_work.eq(1) | weeks.gt(0)
    if not coherent.all():
        raise ValueError(
            "ASEC WORKYN/WTEMP/WRK_CK and HRSWK/WKSWORK contradict past-year work status."
        )

    # Force the actual source producer to run; its ordinary idempotent path
    # would trust any already nonconstant column. The original is untouched.
    source = _with_person(
        frame,
        person.drop(columns=list(US_HOURS_WORKED_OUTPUT_COLUMNS), errors="ignore"),
    )
    produced = with_us_hours_worked_inputs(source, seed=seed, time_period=period)
    expected = produced.table("person")[_USUAL_HOURS]
    known = (
        person[_USUAL_HOURS].notna()
        if _USUAL_HOURS in person
        else pd.Series(False, index=person.index)
    )
    if known.any():
        observed = pd.to_numeric(person.loc[known, _USUAL_HOURS], errors="coerce")
        if not np.array_equal(
            observed.to_numpy(dtype=float, na_value=np.nan),
            expected.loc[known].to_numpy(dtype=float, na_value=np.nan),
        ):
            raise ValueError(
                "Stored usual hours contradict raw ASEC HRSWK; recover their source "
                "or transfer lineage instead of replacing observations."
            )
    result = frame
    if not known.all():
        updated = person.copy(deep=True)
        if _USUAL_HOURS not in updated:
            updated[_USUAL_HOURS] = expected.to_numpy(copy=True)
        else:
            updated.loc[~known, _USUAL_HOURS] = expected.loc[~known].to_numpy()
        result = _with_person(frame, updated)
    # Attach only newly derived ASEC cells to the continuing base by stable
    # person IDs. Existing PUF hours and every other input stay untouched.
    prepared_base = original
    if not known.all():
        updated_base = original_person.copy(deep=True)
        if _USUAL_HOURS not in updated_base:
            updated_base[_USUAL_HOURS] = np.nan
        new_values = result.table("person").set_index("person_id")[_USUAL_HOURS]
        destination = (
            updated_base["person_id"].isin(new_values.index)
            & updated_base[_USUAL_HOURS].isna()
        )
        updated_base.loc[destination, _USUAL_HOURS] = (
            updated_base.loc[destination, "person_id"].map(new_values).to_numpy()
        )
        prepared_base = _with_person(original, updated_base)
    return (
        prepared_base,
        result,
        {
            "source_column": "HRSWK",
            "universe_columns": [source_age_column, "WKSWORK", *work_status_columns],
            "excluded_source_universe_rows": excluded_rows,
            "producer": "with_us_hours_worked_inputs",
            "filled_rows": int((~known).sum()),
            "preserved_observed_rows": int(known.sum()),
            "raw_source_complete": True,
            "source_agreement": True,
            "donor_channel": BASE_ASEC_SUPPORT_CHANNEL,
        },
    )


def acs_local_hours_signal_gate(
    frame: Frame,
    *,
    source_null_audit: Sequence[Mapping[str, object]] = (),
) -> GateResult:
    """Require complete nonconstant hours in each origin before default filling.

    Counts at 0 and 40 are diagnostic only. A source-null receipt still fails
    after a consumer fills those cells, even if the resulting column contains
    several values. This gate measures support/coverage; conditional model
    quality and reform acceptance remain separate release requirements.
    """
    person = frame.table("person")
    tag = spine_column("person")
    failures: list[str] = []
    by_spine: dict[str, object] = {}
    if tag not in person or person[tag].isna().any():
        failures.append(f"Missing person origin tags: {tag}.")
    else:
        for spine in (ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE):
            selected = person.loc[person[tag].eq(spine)]
            detail: dict[str, object] = {"rows": len(selected), "columns": {}}
            by_spine[spine] = detail
            if selected.empty:
                failures.append(f"{spine}: no person rows.")
                continue
            for column in US_HOURS_WORKED_POOL_OUTPUT_COLUMNS:
                if column not in selected:
                    failures.append(f"{spine}: missing {column}.")
                    continue
                values = pd.to_numeric(selected[column], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan
                )
                valid = np.isfinite(values) & (values >= 0) & (values <= 99)
                unique = len(np.unique(values[valid]))
                detail["columns"][column] = {
                    "missing_or_invalid_rows": int((~valid).sum()),
                    "unique_values": unique,
                    "zero_rows": int((values == 0).sum()),
                    "forty_rows": int((values == 40).sum()),
                }
                if not valid.all():
                    failures.append(f"{spine}: {column} has missing or invalid rows.")
                if unique < 2:
                    failures.append(f"{spine}: {column} is constant or has no signal.")
        unknown = set(person[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}
        if unknown:
            failures.append("Local hours origin tags contain an unsupported spine.")
    for entry in source_null_audit:
        if entry.get("entity") == "person" and entry.get("column") in (
            US_HOURS_WORKED_POOL_OUTPUT_COLUMNS
        ):
            missing = entry.get("missing_rows")
            if type(missing) is not int or missing != 0:
                failures.append(
                    f"{entry.get('column')}: unresolved source rows before consumer "
                    "filling; default-filled cells are not transferred hours."
                )
    return GateResult(
        name="acs_local_hours_signal",
        passed=not failures,
        failures=tuple(failures),
        details={"per_spine": by_spine},
    )
