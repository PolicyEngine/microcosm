"""Measured Census SPM independence role as a person input leaf.

``is_spm_independent_minor_role`` is the one dataset source input the engine
declares (``policyengine_us.spm.DATASET_SOURCE_INPUTS``): a Boolean the
producer must deliver from the source, never synthesize. spm-calculator's
measurement classifies a 15-to-17-year-old as an adult only when it is True,
and one SPM unit with no classified adult refuses the whole population's SPM
measurement (``SPM_COMPOSITION_REQUIRED``).

None of the frozen ``census_cps_*.h5`` inputs carries ``SPM_HEAD``, and only the
2024 vintage carries ``A_FAMTYP``/``A_FAMREL``, so the role cannot be read off
the pooled frame. This stage restores it the way ``LKWEEKS``, ``ED_VAL`` and
``PAW_TYP`` are restored: from the pinned complete Census ASEC person CSVs by
exact ``(source_year, PERIDNUM)`` identity, through the certified derivation
:func:`~microcosm.build.us_runtime.spm_role_source.derive_spm_role_source`,
unchanged. That derivation refuses a person with no ASEC origin, a native SPM
unit that repeats a source person or mixes two source units, a unit with more
or fewer than one ``SPM_HEAD``, a unit left with no classified adult after the
role, and any disagreement with Census's own ``SPM_NUMADULTS`` /
``SPM_NUMKIDS`` / ``SPM_NUMPER``. Nothing is imputed, no weight is used, and no
age is changed.

The stage projects the frame's identity, age and Census count columns into a
temporary H5 solely so the derivation can hash and read its input the way it
does for a published parent; the projection's digest is recorded in the
provenance as ``frame_projection_sha256``.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
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
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS,
    fetch_asec_education_assistance_source,
)
from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_role_source import (
    _OPTIONAL_RAW_CHECKS,
    _REQUIRED_RAW_CHECKS,
    ASEC_SPM_ROLE_SOURCES,
    NATIVE_SPM_ROLE,
    AsecSpmRoleSource,
    derive_spm_role_source,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "US_SPM_INDEPENDENCE_ROLE_NONCONSTANT_PERSON_COLUMNS",
    "US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND",
    "US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS",
    "US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS",
    "US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY",
    "US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS",
    "US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY",
    "US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY",
    "US_SPM_INDEPENDENCE_ROLE_STAGE_NAME",
    "derive_us_spm_independence_role_from_manifest",
    "resolve_asec_spm_role_source_paths",
    "us_spm_independence_role_signal_gate",
    "us_spm_independence_role_stage_spec",
    "us_spm_independence_role_summary",
    "with_us_spm_independence_role",
]

US_SPM_INDEPENDENCE_ROLE_STAGE_NAME = "spm_independence_role"
US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND = "derive_spm_independence_role"

US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS: tuple[str, ...] = (NATIVE_SPM_ROLE,)
US_SPM_INDEPENDENCE_ROLE_NONCONSTANT_PERSON_COLUMNS = (
    US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS
)

#: The person columns :func:`derive_spm_role_source` requires of its parent:
#: frame identity, the source join key, the model age, the ASEC identity the
#: source join is verified against, and the raw age/count fields it reconciles
#: against the pinned Census CSV. The tail is imported from the derivation so
#: the two lists cannot drift; a test pins that the derivation refuses each.
US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_id",
    "person_spm_unit_id",
    "source_year",
    "PERIDNUM",
    "age",
    "source_household_id",
    "source_person_id",
    "source_row_id",
    *_REQUIRED_RAW_CHECKS,
)

#: Raw fields the derivation verifies against the pinned CSV *when the frame
#: carries them*. They are projected through when present and never required.
US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS: tuple[str, ...] = _OPTIONAL_RAW_CHECKS

#: ``SourceRuntimeConfig.extra`` keys the handler reads: income year -> pinned
#: complete Census ASEC person CSV path, and (tests only) income year ->
#: :class:`AsecSpmRoleSource` pins for a synthetic CSV.
US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY = "asec_spm_role_source_paths"
US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY = "asec_spm_role_source_pins"

#: Frame metadata key under which :func:`with_us_spm_independence_role` records
#: the derivation's provenance (reconciliation counts, pinned CSV digests).
US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY = "spm_independence_role"

_PROVENANCE_ATTR = "spm_independence_role_provenance"
_SPM_UNIT_TABLE = "spm_unit"
_SPM_UNIT_ID = "spm_unit_id"
_DERIVE_PARAMETER_KEYS = frozenset()

#: Plausibility bands, measured on the three pinned vintages: the role share
#: among all persons (the SPM head plus spouses; 0.601-0.603 per vintage on the
#: phase-2 base, 0.567 on Build P) and among 15-to-17-year-olds (1.49 %-1.73 %
#: per vintage). See ``docs/us-spm-role-stage.md`` §2 and the committed
#: receipts under ``experiments/``.
_ROLE_SHARE_BAND = (0.40, 0.75)
_MINOR_ROLE_SHARE_BAND = (0.003, 0.06)


def us_spm_independence_role_stage_spec() -> SourceStageSpec:
    """Load the packaged ``spm_independence_role`` stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SPM_INDEPENDENCE_ROLE_STAGE_NAME not in stage_map:
        raise ValueError(
            "US source manifest declares no "
            f"{US_SPM_INDEPENDENCE_ROLE_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_SPM_INDEPENDENCE_ROLE_STAGE_NAME]
    if tuple(spec.outputs) != US_SPM_INDEPENDENCE_ROLE_OUTPUT_COLUMNS:
        raise ValueError(
            f"{US_SPM_INDEPENDENCE_ROLE_STAGE_NAME!r} manifest outputs do not "
            "match the runtime-owned SPM independence role."
        )
    return spec


def resolve_asec_spm_role_source_paths(
    paths: Mapping[int, str | Path] | None,
    *,
    income_years: tuple[int, ...] = ASEC_EDUCATION_ASSISTANCE_INCOME_YEARS,
) -> dict[int, Path]:
    """Return one pinned complete person CSV path per pooled income year.

    ``paths`` uses the ``--asec-education-source INCOME_YEAR=PATH`` vocabulary.
    An income year without a path is fetched from the official Census archive
    and verified against the same pins, exactly as the education sidecar is.
    The derivation re-verifies every CSV's size and SHA-256 itself.
    """

    provided = (
        {} if paths is None else {int(year): Path(p) for year, p in paths.items()}
    )
    unknown = sorted(set(provided) - set(ASEC_SPM_ROLE_SOURCES))
    if unknown:
        raise ValueError(
            "No pinned ASEC SPM role source covers income year(s) "
            f"{unknown}; pinned income years: {sorted(ASEC_SPM_ROLE_SOURCES)}."
        )
    unpinned = sorted(set(income_years) - set(ASEC_SPM_ROLE_SOURCES))
    if unpinned:
        raise ValueError(
            "The pooled income year(s) "
            f"{unpinned} have no pinned ASEC SPM role source; pinned income "
            f"years: {sorted(ASEC_SPM_ROLE_SOURCES)}."
        )
    return {
        year: (
            provided[year].expanduser()
            if year in provided
            else fetch_asec_education_assistance_source(year)
        )
        for year in sorted(income_years)
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def derive_us_spm_independence_role_from_manifest(
    frame: pd.DataFrame | None,
    operation: SourceOperationSpec,
    context: SourceRuntimeContext | None,
) -> pd.DataFrame:
    """Attach the source role through the certified derivation, or fail closed.

    The person table is the stage's primary table; the ``spm_unit`` table and
    the pinned CSV paths come from the runtime context. Every refusal of
    :func:`derive_spm_role_source` surfaces as a :class:`SourceRuntimeError`
    naming this stage; nothing is defaulted.
    """

    if operation.kind != US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND:
        raise SourceRuntimeError(
            "US SPM independence role derivation received unexpected operation "
            f"{operation.kind!r}."
        )
    if frame is None:
        raise SourceRuntimeError(
            "US SPM independence role derivation requires the person table to "
            "be read first."
        )
    unexpected = sorted(set(operation.parameters) - _DERIVE_PARAMETER_KEYS)
    if unexpected:
        raise SourceRuntimeError(
            "US SPM independence role derivation received unsupported "
            f"parameter(s): {unexpected}."
        )
    if context is None:
        raise SourceRuntimeError(
            "US SPM independence role derivation requires the runtime context "
            "(the spm_unit table and the pinned Census ASEC person CSV paths)."
        )
    missing = [
        column
        for column in US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise SourceRuntimeError(
            f"US SPM independence role derivation requires person column(s): {missing}."
        )
    spm_unit = context.read_table(_SPM_UNIT_TABLE)
    if _SPM_UNIT_ID not in spm_unit.columns:
        raise SourceRuntimeError(
            "US SPM independence role derivation requires the spm_unit table to "
            f"carry {_SPM_UNIT_ID!r}."
        )
    paths = context.config.extra.get(US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY)
    if not isinstance(paths, Mapping) or not paths:
        raise SourceRuntimeError(
            "US SPM independence role derivation requires the pinned complete "
            "Census ASEC person CSV paths under config.extra"
            f"[{US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY!r}] (income year -> path)."
        )
    pins = context.config.extra.get(US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY)
    if pins is not None and (
        not isinstance(pins, Mapping)
        or not all(isinstance(pin, AsecSpmRoleSource) for pin in pins.values())
    ):
        raise SourceRuntimeError(
            "US SPM independence role derivation source pins must map income "
            "years to AsecSpmRoleSource."
        )

    columns = [
        *US_SPM_INDEPENDENCE_ROLE_REQUIRED_SOURCE_COLUMNS,
        *(
            column
            for column in US_SPM_INDEPENDENCE_ROLE_OPTIONAL_SOURCE_COLUMNS
            if column in frame.columns
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="spm-independence-role-") as scratch:
        projection = Path(scratch) / "frame_projection.h5"
        frame[columns].to_hdf(projection, key="person", mode="w", format="fixed")
        spm_unit[[_SPM_UNIT_ID]].to_hdf(
            projection, key=_SPM_UNIT_TABLE, mode="a", format="fixed"
        )
        digest = _sha256(projection)
        try:
            result = derive_spm_role_source(
                projection,
                {int(year): Path(path) for year, path in paths.items()},
                expected_parent_sha256=digest,
                source_pins=None if pins is None else dict(pins),
            )
        except ValueError as error:
            raise SourceRuntimeError(
                f"US SPM independence role derivation refused: {error}"
            ) from error

    if not np.array_equal(
        result.evidence["person_id"].to_numpy(), frame["person_id"].to_numpy()
    ):
        raise SourceRuntimeError(
            "US SPM independence role derivation returned roles out of person order."
        )
    role = np.asarray(result.role, dtype=bool)
    if role.shape != (len(frame),):
        raise SourceRuntimeError(
            "US SPM independence role derivation did not cover every person."
        )
    output = frame.copy(deep=True)
    output[NATIVE_SPM_ROLE] = role
    output.attrs[_PROVENANCE_ATTR] = {
        **result.provenance,
        "frame_projection_sha256": digest,
        "frame_projection_columns": list(columns),
    }
    return output


def _role_surface_carries_signal(frame: Frame) -> bool:
    person = frame.table("person")
    if NATIVE_SPM_ROLE not in person:
        return False
    return person[NATIVE_SPM_ROLE].dropna().nunique() > 1


def _frame_income_years(person: pd.DataFrame) -> tuple[int, ...]:
    years = pd.to_numeric(person["source_year"], errors="coerce")
    if years.isna().any() or (years != np.floor(years)).any():
        raise ValueError("US SPM independence role requires integer source_year.")
    return tuple(sorted(int(year) for year in years.unique()))


def with_us_spm_independence_role(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    asec_spm_role_source_paths: Mapping[int, str | Path] | None = None,
    source_pins: Mapping[int, AsecSpmRoleSource] | None = None,
) -> Frame:
    """Materialize the measured SPM independence role on a US frame.

    A frame already carrying a non-degenerate role column is returned as is.
    ``asec_spm_role_source_paths`` maps income years to the pinned complete
    Census ASEC person CSVs; years without a path are fetched and verified.
    ``source_pins`` exists for synthetic tests and defaults to the certified
    pins of the frame's own income years.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US SPM independence role requires the US schema.")
    if _role_surface_carries_signal(frame):
        return frame

    person = frame.table("person")
    if "source_year" not in person:
        raise ValueError(
            "US SPM independence role requires the person table to carry source_year."
        )
    income_years = _frame_income_years(person)
    if source_pins is None:
        paths = resolve_asec_spm_role_source_paths(
            asec_spm_role_source_paths, income_years=income_years
        )
        pins = {year: ASEC_SPM_ROLE_SOURCES[year] for year in income_years}
    else:
        if asec_spm_role_source_paths is None:
            raise ValueError(
                "US SPM independence role requires explicit CSV paths when "
                "source pins are supplied."
            )
        paths = {
            int(year): Path(path).expanduser()
            for year, path in asec_spm_role_source_paths.items()
        }
        pins = dict(source_pins)

    output = run_source_stage(
        us_spm_independence_role_stage_spec(),
        tables={
            "person": person.copy(deep=True),
            _SPM_UNIT_TABLE: frame.table(_SPM_UNIT_TABLE).copy(deep=True),
        },
        operation_handlers={
            US_SPM_INDEPENDENCE_ROLE_OPERATION_KIND: (
                derive_us_spm_independence_role_from_manifest
            )
        },
        config=SourceRuntimeConfig(
            seed=int(seed),
            target_year=int(time_period),
            extra={
                US_SPM_INDEPENDENCE_ROLE_SOURCE_PATHS_KEY: paths,
                US_SPM_INDEPENDENCE_ROLE_SOURCE_PINS_KEY: pins,
            },
        ),
    )
    aligned = output.set_index("person_id").reindex(person["person_id"])
    if aligned[NATIVE_SPM_ROLE].isna().any():
        raise ValueError(
            "US SPM independence role stage output does not cover every person "
            f"for {NATIVE_SPM_ROLE!r}."
        )
    provenance = output.attrs.get(_PROVENANCE_ATTR)
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "US SPM independence role stage returned no derivation provenance."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][NATIVE_SPM_ROLE] = aligned[NATIVE_SPM_ROLE].to_numpy(dtype=bool)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY: _json_ready(provenance),
        },
    )


def _json_ready(value: Any) -> Any:
    """Frame metadata must be plain, hashable-friendly builtins."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(nested) for key, nested in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def us_spm_independence_role_summary(frame: Frame) -> dict[str, object]:
    """Return role shares, the composition verdict, and the recorded provenance."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    role = person[NATIVE_SPM_ROLE].fillna(False).astype(bool).to_numpy()
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    minor = (age >= 15.0) & (age < 18.0)
    minor_weight = float(weights[minor].sum())

    composition = check_spm_composition(frame)
    provenance = frame.metadata.get(US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY)
    return {
        "role_share": (
            float(weights[role].sum()) / total_weight if total_weight else 0.0
        ),
        "role_share_band": list(_ROLE_SHARE_BAND),
        "minor_role_share": (
            float(weights[role & minor].sum()) / minor_weight if minor_weight else 0.0
        ),
        "minor_role_share_band": list(_MINOR_ROLE_SHARE_BAND),
        "persons_aged_15_to_17": int(minor.sum()),
        "independent_minor_persons": int((role & minor).sum()),
        "role_missing_values": int(person[NATIVE_SPM_ROLE].isna().sum()),
        "unique_counts": {
            NATIVE_SPM_ROLE: int(person[NATIVE_SPM_ROLE].dropna().nunique())
        },
        "spm_composition": {
            "status": composition.status,
            "role_source": composition.details["role_source"],
            "n_units": composition.details["n_units"],
            "n_units_without_classified_adult": composition.details[
                "n_units_without_classified_adult"
            ],
            "n_units_without_member_aged_18_or_over": composition.details[
                "n_units_without_member_aged_18_or_over"
            ],
        },
        "derivation": None if provenance is None else dict(provenance),
    }


def us_spm_independence_role_signal_gate(frame: Frame) -> GateResult:
    """Require a source-delivered, non-degenerate role that classifies every unit."""

    person = frame.table("person")
    if NATIVE_SPM_ROLE not in person:
        return GateResult(
            name="spm_independence_role_signal",
            passed=False,
            failures=(f"person columns missing: [{NATIVE_SPM_ROLE!r}].",),
            details={"missing": [NATIVE_SPM_ROLE]},
        )

    summary = us_spm_independence_role_summary(frame)
    failures: list[str] = []
    values = person[NATIVE_SPM_ROLE].dropna()
    if not values.isin((True, False)).all():
        failures.append(f"{NATIVE_SPM_ROLE} carries non-Boolean values.")
    missing_values = int(summary["role_missing_values"])
    if missing_values:
        failures.append(
            f"{NATIVE_SPM_ROLE} is missing for {missing_values} person(s); the "
            "engine reads a stored role, never a default."
        )
    if int(summary["unique_counts"][NATIVE_SPM_ROLE]) < 2:
        failures.append(
            f"{NATIVE_SPM_ROLE} is degenerate with "
            f"{summary['unique_counts'][NATIVE_SPM_ROLE]} distinct value(s)."
        )
    composition = summary["spm_composition"]
    if composition["role_source"] != "source_column":
        failures.append(
            "SPM composition is not classified from the source column "
            f"(role source: {composition['role_source']})."
        )
    unresolved = int(composition["n_units_without_classified_adult"])
    if unresolved:
        failures.append(
            f"{unresolved} SPM unit(s) have no classified adult after the role; "
            "the whole population's SPM measurement would raise "
            "SPM_COMPOSITION_REQUIRED."
        )
    for share_key, band_key, label in (
        ("role_share", "role_share_band", "SPM independence role weighted share"),
        (
            "minor_role_share",
            "minor_role_share_band",
            "SPM independence role weighted share among 15-to-17-year-olds",
        ),
    ):
        share = float(summary[share_key])
        lower, upper = summary[band_key]
        if not lower <= share <= upper:
            failures.append(f"{label} {share:.6f} outside [{lower:.6f}, {upper:.6f}].")
    return GateResult(
        name="spm_independence_role_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
