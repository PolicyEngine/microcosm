"""Build-stage orchestration for the licensed NSECE attendance source extension.

The existing generation-0 source manifest is byte-frozen. This extension has a
separate packaged SourceStageSpec and runs after relationship/hours producers.
It is also callable on an exact native parent for candidate qualification.
"""

from __future__ import annotations

import json
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import canonicalize_table_string_dtypes
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    childcare_attendance_contract,
)
from microcosm.build.us_runtime.childcare_population import (
    harmonize_asec_childcare_predictors,
)
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CHILDCARE_FALLBACK_COLUMNS,
    NSECE_CHILDCARE_MATCH_COLUMNS,
    assert_childcare_attendance_exportable,
    load_nsece_childcare,
    with_us_nsece_childcare_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_bridge import (
    bridge_nsece_noncalendar_attendance,
)
from microcosm.build.us_runtime.nsece_childcare_dependence import (
    fit_nsece_sibling_dependence,
)
from microcosm.frame import Frame


def inherit_outside_domain_attendance_baseline(frame: Frame) -> Frame:
    """Explicit export policy: retain engine baseline only outside modeled ages.

    This does NOT infer measured nonattendance for teens or disabled older
    children. Receipts identify the inherited values and source-domain limits.
    Observed values are preserved and all under-13 records must be resolved.
    """
    from policyengine_us import CountryTaxBenefitSystem

    people = frame.table("person").copy()
    system = CountryTaxBenefitSystem()
    outside = ~people.age.between(0, 12)
    defaults, inherited = {}, {}
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        if column not in people:
            raise ValueError(
                "Attendance must be imputed before inheriting the outside-domain baseline."
            )
        if people.loc[~outside, column].isna().any():
            raise ValueError(
                "Cannot inherit baseline for unresolved under-13 attendance."
            )
        default = system.variables[column].default_value
        if not np.isfinite(default):
            raise ValueError("The pinned engine attendance baseline must be finite.")
        missing = outside & people[column].isna()
        people.loc[missing, column] = default
        people.loc[missing, f"{column}_source"] = (
            "inherited_engine_baseline_outside_age_0_12"
        )
        defaults[column] = float(default)
        inherited[column] = int(missing.sum())
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = people
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            "childcare_outside_domain_baseline": {
                "engine_version": version("policyengine-us"),
                "values": defaults,
                "inherited_counts": inherited,
                "interpretation": "baseline retained; not evidence of nonattendance outside ages 0-12",
            },
        },
    )


def with_us_childcare_attendance_inputs(
    frame: Frame,
    *,
    household_tsv: str | Path,
    calendar_tsv: str | Path,
    asec_source_cache: str | Path | None,
    seed: int,
    inherit_outside_domain_baseline: bool = False,
) -> Frame:
    """Run the source/bridge/target recipe and enforce the export contract."""
    contract = childcare_attendance_contract()
    spec = SourceStageSpec.from_mapping(contract)
    if tuple(spec.outputs) != US_CHILDCARE_ATTENDANCE_COLUMNS:
        raise ValueError(
            "Childcare source-stage outputs drifted from the engine inputs."
        )
    source = load_nsece_childcare(household_tsv, calendar_tsv)
    dependence = fit_nsece_sibling_dependence(source.children)
    source = bridge_nsece_noncalendar_attendance(source, seed=seed)
    normalized = harmonize_asec_childcare_predictors(
        frame, source_cache=asec_source_cache
    )
    candidate = with_us_nsece_childcare_attendance(
        normalized,
        source,
        seed=seed,
        match_columns=NSECE_CHILDCARE_MATCH_COLUMNS,
        fallback_match_columns=NSECE_CHILDCARE_FALLBACK_COLUMNS,
        sibling_dependence=dependence["rho"],
    )
    if inherit_outside_domain_baseline:
        candidate = inherit_outside_domain_attendance_baseline(candidate)
    assert_childcare_attendance_exportable(candidate)
    return Frame(
        {entity: candidate.table(entity) for entity in candidate.entities},
        candidate.schema,
        {
            entity: candidate.weights_for(entity)
            for entity in candidate.weighted_entities
        },
        candidate.strata,
        mass_log=candidate.mass_log,
        metadata={
            **candidate.metadata,
            "childcare_attendance_stage": {
                "stage": spec.stage,
                "outputs": spec.outputs,
                "seed": seed,
                "operation_order": tuple(
                    operation.kind for operation in spec.operations
                ),
                "sibling_dependence": dependence,
                "modeled_age_domain": [0, 12],
                "outside_domain_policy": "inherit_engine_baseline"
                if inherit_outside_domain_baseline
                else "require_observed",
            },
        },
    )


def export_native_childcare_candidate(
    parent_path: str | Path, candidate: Frame, output_path: str | Path
) -> Path:
    """Add only attendance inputs to a new native H5, verifying every old column.

    The parent and its period are preserved. This is a local candidate export,
    not a source-enrichment release or a publication/certification operation.
    """
    from policyengine_us.data import USSingleYearDataset

    parent_path, output_path = Path(parent_path), Path(output_path)
    if output_path.exists() or parent_path.resolve() == output_path.resolve():
        raise ValueError("Native childcare output must be a new path.")
    assert_childcare_attendance_exportable(candidate)
    parent = USSingleYearDataset(file_path=str(parent_path))
    attendance = set(US_CHILDCARE_ATTENDANCE_COLUMNS)
    for entity in candidate.entities:
        old = getattr(parent, entity)
        reference = old.drop(columns=["household_weight"], errors="ignore")
        columns = [c for c in reference if c not in attendance]
        pd.testing.assert_frame_equal(
            canonicalize_table_string_dtypes(
                reference[columns],
                boundary="native_childcare_export",
                table_name=entity,
            ),
            canonicalize_table_string_dtypes(
                candidate.table(entity)[columns],
                boundary="native_childcare_export",
                table_name=entity,
            ),
        )
    np.testing.assert_array_equal(
        parent.household.household_weight.to_numpy(),
        candidate.weights_for("household").values,
    )
    people = parent.person.copy()
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        people[column] = candidate.table("person")[column]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial.h5")
    if temporary.exists():
        raise ValueError("Native childcare temporary path already exists.")
    try:
        shutil.copyfile(parent_path, temporary)
        with pd.HDFStore(temporary, mode="a") as store:
            store.put("person", people, format="table", data_columns=True)
            store.put(
                "_childcare_attendance_receipt",
                pd.Series(
                    [json.dumps(candidate.metadata, default=dict, allow_nan=False)]
                ),
            )
        loaded = USSingleYearDataset(file_path=str(temporary))
        pd.testing.assert_frame_equal(loaded.person, people)
        for entity in candidate.schema.group_entities:
            pd.testing.assert_frame_equal(
                getattr(loaded, entity), getattr(parent, entity)
            )
        if loaded.time_period != parent.time_period:
            raise ValueError("Native childcare export changed the parent's period.")
        temporary.rename(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
