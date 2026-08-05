"""Dataset-level UK row-wise geography assignment.

This module ties the row-wise household geography primitives to the
PolicyEngine-UK single-year table contract. It deliberately works with explicit
frames or H5 files and does not import an incumbent data package.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.uk_runtime.geography_ladder import (
    UK_GEOGRAPHY_LADDER_COLUMNS,
    UkOaLadder,
    assign_uk_geography_ladder,
    uk_geography_ladder_gate,
)
from populace.build.uk_runtime.national_build import (
    _mass_log_from_stored,
    _read_weight_metadata,
    _weight_kind_from_stored,
    _write_weight_metadata,
)
from populace.build.uk_runtime.rowwise_geography import (
    ROWWISE_GEOGRAPHY_COLUMNS,
    RowwiseGeographyAssignment,
    _source_household_keys,
    assign_household_geography,
    clone_entity_frame,
    id_multiplier_for_values,
)
from populace.frame import MassChangeRecord, WeightKind

#: Declared bound for the clone operator's exact mass conservation: dividing
#: each household weight by ``n_clones`` and duplicating rows changes the
#: total only by float summation error, so anything beyond this relative
#: tolerance is a defect, never an acceptable drift.
MASS_CONSERVATION_RELATIVE_TOLERANCE = 1e-9

#: Pool-grain lineage layer derived by ``apply_uk_source_lineage_modulus``.
#: Distinct from the immediate-layer ``source_household_id`` the national
#: staging H5 already carries.
POOL_SOURCE_LINEAGE_COLUMN = "pool_source_household_id"

PERSON_ID_COLUMNS = (
    "person_id",
    "person_household_id",
    "person_benunit_id",
)
BENUNIT_ID_COLUMNS = ("benunit_id",)
HOUSEHOLD_ID_COLUMNS = ("household_id",)
UK_SINGLE_YEAR_TABLES = ("person", "benunit", "household", "time_period")


@dataclass(frozen=True)
class UKRowwiseDatasetResult:
    """Cloned UK single-year tables and row-wise geography metadata.

    ``household_weight_kind`` and ``mass_log`` carry the national seam's
    weight provenance through the clone. Absence on the input defaults to
    ``WeightKind.DESIGN`` — the same semantics the national loader applies to
    an attr-less H5 — and the clone always appends one mass-conserving record
    documenting the ``n_clones`` split.
    """

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    assignment: RowwiseGeographyAssignment
    time_period: str
    output_path: Path | None = None
    household_weight_kind: WeightKind = WeightKind.DESIGN
    mass_log: tuple[MassChangeRecord, ...] = ()

    @property
    def id_multiplier(self) -> int:
        return self.assignment.id_multiplier

    @property
    def n_clones(self) -> int:
        return self.assignment.n_clones


def clone_uk_dataset_tables_with_rowwise_geography(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    crosswalk: pd.DataFrame,
    n_clones: int = 1,
    seed: int = 42,
    time_period: int | str | None = None,
    source_year: int | None = None,
    id_multiplier: int | None = None,
    household_id_column: str = "household_id",
    household_weight_column: str = "household_weight",
    country_column: str | None = None,
    region_column: str = "region",
    clone_index_column: str | None = "clone_index",
    require_all_countries: bool = True,
    require_constituency: bool = True,
    constrain_to_region: bool = True,
    allow_zero_population_distribution: bool = False,
    avoid_constituency_collisions: bool = True,
    household_weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
    source_lineage_modulus: int | None = None,
) -> UKRowwiseDatasetResult:
    """Clone UK person/benunit/household tables and assign row-wise geography."""

    _validate_weight_metadata(household_weight_kind, mass_log)
    person_frame = person.copy()
    benunit_frame = benunit.copy()
    household_frame = household.copy()
    _validate_input_tables(
        person_frame,
        benunit_frame,
        household_frame,
        household_id_column=household_id_column,
        household_weight_column=household_weight_column,
    )
    if source_lineage_modulus is not None:
        household_frame = apply_uk_source_lineage_modulus(
            household_frame,
            modulus=source_lineage_modulus,
            household_id_column=household_id_column,
        )
    input_total = float(
        np.asarray(
            pd.to_numeric(household_frame[household_weight_column], errors="raise"),
            dtype=np.float64,
        ).sum()
    )
    if not np.isfinite(input_total):
        raise ValueError("household weight total must be finite before cloning.")
    if input_total <= 0.0:
        raise ValueError(
            "household weights must carry positive total mass before cloning; "
            "an all-zero pool would clone to a dataset the national loader "
            "rejects."
        )
    _assert_mass_log_current(mass_log, input_total)

    if id_multiplier is None:
        id_multiplier = id_multiplier_for_values(
            household_frame[household_id_column],
            person_frame["person_id"],
            person_frame["person_household_id"],
            person_frame["person_benunit_id"],
            benunit_frame["benunit_id"],
        )
    assignment = assign_household_geography(
        household_frame,
        crosswalk,
        n_clones=n_clones,
        seed=seed,
        id_multiplier=id_multiplier,
        household_id_column=household_id_column,
        weight_column=household_weight_column,
        country_column=country_column,
        region_column=region_column,
        source_year=source_year,
        require_all_countries=require_all_countries,
        require_constituency=require_constituency,
        constrain_to_region=constrain_to_region,
        allow_zero_population_distribution=allow_zero_population_distribution,
        avoid_constituency_collisions=avoid_constituency_collisions,
    )

    cloned_person = clone_entity_frame(
        person_frame,
        id_columns=PERSON_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=assignment.id_multiplier,
        clone_index_column=clone_index_column,
    )
    cloned_benunit = clone_entity_frame(
        benunit_frame,
        id_columns=BENUNIT_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=assignment.id_multiplier,
        clone_index_column=clone_index_column,
    )
    output_total = float(
        np.asarray(
            assignment.household["household_weight"],
            dtype=np.float64,
        ).sum()
    )
    _assert_household_mass_conserved(input_total, output_total)
    clone_record = MassChangeRecord(
        entity="household",
        old_total=input_total,
        new_total=output_total,
        declared_factor=1.0,
        reason=(
            f"Rowwise geography clone at n_clones={n_clones} divides each "
            f"household weight by {n_clones}; total household mass is "
            "conserved."
        ),
    )
    result = UKRowwiseDatasetResult(
        person=cloned_person.reset_index(drop=True),
        benunit=cloned_benunit.reset_index(drop=True),
        household=assignment.household.reset_index(drop=True),
        assignment=assignment,
        time_period=_normalise_time_period(time_period, source_year=source_year),
        household_weight_kind=household_weight_kind,
        mass_log=(*mass_log, clone_record),
    )
    validate_uk_rowwise_dataset_tables(result.person, result.benunit, result.household)
    return result


@dataclass(frozen=True)
class UKLadderRowwiseDatasetResult:
    """Cloned UK tables with OA-ladder geography and the passed release gate.

    The ladder route is the release path (#495 increment 6a): geography comes
    from :func:`assign_uk_geography_ladder` under the artifact's vintage
    discipline, the release-blocking :func:`uk_geography_ladder_gate` must
    pass before a result exists, and the #501 weight-kind/mass-log fence
    chain carries unchanged. Declared design delta vs the crosswalk route:
    no cross-clone constituency collision avoidance — duplicate
    (source, constituency) pairs are a reported diagnostic.
    """

    person: pd.DataFrame
    benunit: pd.DataFrame
    household: pd.DataFrame
    time_period: str
    gate: GateResult
    n_clones: int
    id_multiplier: int
    output_path: Path | None = None
    household_weight_kind: WeightKind = WeightKind.DESIGN
    mass_log: tuple[MassChangeRecord, ...] = ()


def clone_uk_dataset_tables_with_ladder_geography(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    ladder: UkOaLadder,
    n_clones: int = 1,
    seed: int = 42,
    time_period: int | str | None = None,
    source_year: int | None = None,
    id_multiplier: int | None = None,
    clone_index_column: str | None = "clone_index",
    expected_constituency_vintage: str | None = None,
    region_column: str = "region",
    household_weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
    source_lineage_modulus: int | None = None,
) -> UKLadderRowwiseDatasetResult:
    """Clone UK tables and assign geography through the OA ladder."""

    _validate_weight_metadata(household_weight_kind, mass_log)
    person_frame = person.copy()
    benunit_frame = benunit.copy()
    household_frame = household.copy()
    _validate_input_tables(
        person_frame,
        benunit_frame,
        household_frame,
        household_id_column="household_id",
        household_weight_column="household_weight",
    )
    if source_lineage_modulus is not None:
        household_frame = apply_uk_source_lineage_modulus(
            household_frame,
            modulus=source_lineage_modulus,
        )
    household_frame = _attach_source_lineage(household_frame, source_year=source_year)
    weight_values = np.asarray(
        pd.to_numeric(household_frame["household_weight"], errors="raise"),
        dtype=np.float64,
    )
    if not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError(
            "household weights must be finite and non-negative before "
            "cloning; a negative component cannot hide behind a positive "
            "aggregate."
        )
    input_total = float(weight_values.sum())
    if input_total <= 0.0:
        raise ValueError(
            "household weights must carry positive total mass before cloning."
        )
    _assert_mass_log_current(mass_log, input_total)

    if id_multiplier is None:
        id_multiplier = id_multiplier_for_values(
            household_frame["household_id"],
            person_frame["person_id"],
            person_frame["person_household_id"],
            person_frame["person_benunit_id"],
            benunit_frame["benunit_id"],
        )
    cloned_household = clone_entity_frame(
        household_frame,
        id_columns=HOUSEHOLD_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=id_multiplier,
        clone_index_column=clone_index_column,
    ).reset_index(drop=True)
    cloned_household["household_weight"] = (
        np.asarray(cloned_household["household_weight"], dtype=np.float64) / n_clones
    )
    cloned_person = clone_entity_frame(
        person_frame,
        id_columns=PERSON_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=id_multiplier,
        clone_index_column=clone_index_column,
    ).reset_index(drop=True)
    cloned_benunit = clone_entity_frame(
        benunit_frame,
        id_columns=BENUNIT_ID_COLUMNS,
        n_clones=n_clones,
        id_multiplier=id_multiplier,
        clone_index_column=clone_index_column,
    ).reset_index(drop=True)

    if clone_index_column is not None:
        _assert_clone_link_alignment(
            cloned_person,
            cloned_household,
            clone_index_column=clone_index_column,
        )

    assigned = assign_uk_geography_ladder(
        cloned_household,
        ladder,
        seed=seed,
        expected_constituency_vintage=expected_constituency_vintage,
        region_column=region_column,
    ).reset_index(drop=True)

    output_total = float(
        np.asarray(assigned["household_weight"], dtype=np.float64).sum()
    )
    _assert_household_mass_conserved(input_total, output_total)
    clone_record = MassChangeRecord(
        entity="household",
        old_total=input_total,
        new_total=output_total,
        declared_factor=1.0,
        reason=(
            f"Rowwise ladder clone at n_clones={n_clones} divides each "
            f"household weight by {n_clones}; total household mass is "
            "conserved."
        ),
    )

    gate = uk_geography_ladder_gate(
        assigned,
        np.asarray(assigned["household_weight"], dtype=np.float64),
        region_column=region_column,
    )
    if not gate.passed:
        raise ValueError(
            "UK geography ladder gate failed on the cloned assignment: "
            + "; ".join(gate.failures)
        )

    result = UKLadderRowwiseDatasetResult(
        person=cloned_person,
        benunit=cloned_benunit,
        household=assigned,
        time_period=_normalise_time_period(time_period, source_year=source_year),
        gate=gate,
        n_clones=n_clones,
        id_multiplier=id_multiplier,
        household_weight_kind=household_weight_kind,
        mass_log=(*mass_log, clone_record),
    )
    validate_uk_ladder_rowwise_dataset_tables(
        result.person,
        result.benunit,
        result.household,
    )
    return result


def clone_uk_dataset_with_ladder_geography(
    dataset: Any | str | Path,
    ladder: UkOaLadder,
    *,
    output_path: str | Path | None = None,
    n_clones: int = 1,
    seed: int = 42,
    source_year: int | None = None,
    id_multiplier: int | None = None,
    clone_index_column: str | None = "clone_index",
    expected_constituency_vintage: str | None = None,
    region_column: str = "region",
    source_lineage_modulus: int | None = None,
) -> UKLadderRowwiseDatasetResult:
    """Clone a UK dataset object or H5 with OA-ladder geography.

    Weight kind and mass log come from the input exactly as in the crosswalk
    route (absence keeps the national loader's DESIGN semantics; unknown
    kinds fail closed).
    """

    tables = _dataset_tables(dataset, source_year=source_year)
    result = clone_uk_dataset_tables_with_ladder_geography(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        ladder=ladder,
        n_clones=n_clones,
        seed=seed,
        time_period=tables["time_period"],
        source_year=source_year,
        id_multiplier=id_multiplier,
        clone_index_column=clone_index_column,
        expected_constituency_vintage=expected_constituency_vintage,
        region_column=region_column,
        household_weight_kind=tables["household_weight_kind"],
        mass_log=tables["mass_log"],
        source_lineage_modulus=source_lineage_modulus,
    )
    if output_path is None:
        return result
    path = write_uk_rowwise_dataset(result, output_path)
    return UKLadderRowwiseDatasetResult(
        person=result.person,
        benunit=result.benunit,
        household=result.household,
        time_period=result.time_period,
        gate=result.gate,
        n_clones=result.n_clones,
        id_multiplier=result.id_multiplier,
        output_path=path,
        household_weight_kind=result.household_weight_kind,
        mass_log=result.mass_log,
    )


def validate_uk_ladder_rowwise_dataset_tables(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    """Validate entity IDs, links, and ladder geography columns."""

    _require_columns(person, PERSON_ID_COLUMNS, label="person")
    _require_columns(benunit, BENUNIT_ID_COLUMNS, label="benunit")
    _require_columns(household, HOUSEHOLD_ID_COLUMNS, label="household")
    _require_columns(household, UK_GEOGRAPHY_LADDER_COLUMNS, label="household")
    for column in UK_GEOGRAPHY_LADDER_COLUMNS:
        values = household[column].fillna("").astype(str).str.strip()
        blank = values == ""
        if blank.any():
            raise ValueError(
                f"household.{column} contains {int(blank.sum())} blank "
                "value(s); every ladder rung must be assigned."
            )

    _require_unique(person, "person_id", label="person")
    _require_unique(benunit, "benunit_id", label="benunit")
    _require_unique(household, "household_id", label="household")

    household_ids = set(household["household_id"])
    missing_households = sorted(set(person["person_household_id"]) - household_ids)
    if missing_households:
        raise ValueError(
            "person.person_household_id contains value(s) absent from household: "
            f"{missing_households[:5]}."
        )
    benunit_ids = set(benunit["benunit_id"])
    missing_benunits = sorted(set(person["person_benunit_id"]) - benunit_ids)
    if missing_benunits:
        raise ValueError(
            "person.person_benunit_id contains value(s) absent from benunit: "
            f"{missing_benunits[:5]}."
        )


def _assert_clone_link_alignment(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    clone_index_column: str,
) -> None:
    """Refuse cross-clone links an undersized explicit id_multiplier allows.

    Set-membership FK validation cannot see a clone-1 person pointing at a
    clone-0 household when remapped ids collide; the clone indices must
    agree row by row.
    """

    household_clone = household.set_index("household_id")[clone_index_column]
    mapped = person["person_household_id"].map(household_clone)
    misaligned = mapped.to_numpy() != person[clone_index_column].to_numpy()
    if misaligned.any():
        raise ValueError(
            f"{int(misaligned.sum())} person row(s) link across clone "
            "generations; id_multiplier is too small for these ids."
        )


def _attach_source_lineage(
    household: pd.DataFrame,
    *,
    source_year: int | None,
) -> pd.DataFrame:
    """Default the immediate-layer lineage columns the crosswalk route adds."""

    frame = household
    if "source_household_id" not in frame.columns:
        frame["source_household_id"] = frame["household_id"]
    if "source_year" not in frame.columns and source_year is not None:
        frame["source_year"] = source_year
    if "source_household_key" not in frame.columns:
        frame["source_household_key"] = _source_household_keys(
            frame["source_year"] if "source_year" in frame.columns else None,
            frame["source_household_id"],
            source_year=source_year,
        )
    return frame


def clone_uk_dataset_with_rowwise_geography(
    dataset: Any | str | Path,
    crosswalk: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
    n_clones: int = 1,
    seed: int = 42,
    source_year: int | None = None,
    id_multiplier: int | None = None,
    country_column: str | None = None,
    region_column: str = "region",
    clone_index_column: str | None = "clone_index",
    require_all_countries: bool = True,
    require_constituency: bool = True,
    constrain_to_region: bool = True,
    allow_zero_population_distribution: bool = False,
    avoid_constituency_collisions: bool = True,
    source_lineage_modulus: int | None = None,
) -> UKRowwiseDatasetResult:
    """Clone a UK single-year dataset object or H5 path with row-wise geography.

    The input's stored weight kind and mass log are carried, never overridden:
    an H5 supplies them via the national metadata attrs (absence means
    ``WeightKind.DESIGN`` and an empty log, exactly as the national loader
    reads it), and a dataset object supplies them via equally named
    attributes when present.
    """

    tables = _dataset_tables(dataset, source_year=source_year)
    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        crosswalk=crosswalk,
        n_clones=n_clones,
        seed=seed,
        time_period=tables["time_period"],
        source_year=source_year,
        id_multiplier=id_multiplier,
        country_column=country_column,
        region_column=region_column,
        clone_index_column=clone_index_column,
        require_all_countries=require_all_countries,
        require_constituency=require_constituency,
        constrain_to_region=constrain_to_region,
        allow_zero_population_distribution=allow_zero_population_distribution,
        avoid_constituency_collisions=avoid_constituency_collisions,
        household_weight_kind=tables["household_weight_kind"],
        mass_log=tables["mass_log"],
        source_lineage_modulus=source_lineage_modulus,
    )
    if output_path is None:
        return result
    path = write_uk_rowwise_dataset(result, output_path)
    return UKRowwiseDatasetResult(
        person=result.person,
        benunit=result.benunit,
        household=result.household,
        assignment=result.assignment,
        time_period=result.time_period,
        output_path=path,
        household_weight_kind=result.household_weight_kind,
        mass_log=result.mass_log,
    )


def write_uk_rowwise_dataset(
    result: UKRowwiseDatasetResult,
    output_path: str | Path,
) -> Path:
    """Write cloned UK row-wise tables as a valid single-year H5 dataset."""

    # A frozen dataclass does not freeze DataFrames: re-verify the mass log
    # against the tables actually being written, so a post-clone mutation
    # cannot ship under a stale conservation record — and for ladder results
    # (which carry a gate), re-run the release gate on the frame actually
    # written, so a post-gate geography mutation cannot ship either.
    if isinstance(getattr(result, "gate", None), GateResult):
        regate = uk_geography_ladder_gate(
            result.household,
            np.asarray(result.household["household_weight"], dtype=np.float64),
        )
        if not regate.passed:
            raise ValueError(
                "UK geography ladder gate failed on the frame being written "
                "(mutated after the clone?): " + "; ".join(regate.failures)
            )
    _assert_mass_log_current(
        result.mass_log,
        float(np.asarray(result.household["household_weight"], dtype=np.float64).sum()),
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tables and the weight-kind/mass-log attrs must land together: writing
    # them into a temporary file and renaming keeps a metadata failure from
    # leaving a complete-looking attr-less H5 that would silently default to
    # DESIGN semantics on the next read.
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.h5")
    try:
        with pd.HDFStore(temporary_path) as store:
            store.put("person", result.person, format="table", data_columns=True)
            store.put("benunit", result.benunit, format="table", data_columns=True)
            store.put(
                "household",
                result.household,
                format="table",
                data_columns=True,
            )
            store.put(
                "time_period",
                pd.Series([result.time_period]),
                format="table",
                data_columns=True,
            )
        # The national seam's own writer supplies the attrs so the rowwise
        # output is self-describing under ``load_uk_national_dataset``.
        _write_weight_metadata(temporary_path, result)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def validate_uk_rowwise_dataset_tables(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> None:
    """Validate entity IDs and row-wise household geography columns."""

    _require_columns(person, PERSON_ID_COLUMNS, label="person")
    _require_columns(benunit, BENUNIT_ID_COLUMNS, label="benunit")
    _require_columns(household, HOUSEHOLD_ID_COLUMNS, label="household")
    _require_columns(household, ROWWISE_GEOGRAPHY_COLUMNS, label="household")

    _require_unique(person, "person_id", label="person")
    _require_unique(benunit, "benunit_id", label="benunit")
    _require_unique(household, "household_id", label="household")

    household_ids = set(household["household_id"])
    missing_households = sorted(set(person["person_household_id"]) - household_ids)
    if missing_households:
        raise ValueError(
            "person.person_household_id contains value(s) absent from household: "
            f"{missing_households[:5]}."
        )

    benunit_ids = set(benunit["benunit_id"])
    missing_benunits = sorted(set(person["person_benunit_id"]) - benunit_ids)
    if missing_benunits:
        raise ValueError(
            "person.person_benunit_id contains value(s) absent from benunit: "
            f"{missing_benunits[:5]}."
        )


def read_uk_single_year_weight_metadata(
    path: str | Path,
) -> tuple[WeightKind, tuple[MassChangeRecord, ...]]:
    """Read the national weight-kind/mass-log attrs of a UK single-year H5.

    Absence has the national loader's semantics: an attr-less H5 is
    ``WeightKind.DESIGN`` with an empty mass log. An unrecognized stored kind
    fails closed.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"UK single-year dataset not found: {dataset_path}.")
    stored_kind, stored_mass_log = _read_weight_metadata(dataset_path)
    return _weight_kind_from_stored(stored_kind), _mass_log_from_stored(stored_mass_log)


def _dataset_tables(
    dataset: Any | str | Path,
    *,
    source_year: int | None,
) -> dict[str, Any]:
    if isinstance(dataset, str | Path):
        return _read_uk_single_year_h5(dataset)
    missing = [
        name
        for name in ("person", "benunit", "household")
        if not hasattr(dataset, name)
    ]
    if missing:
        raise ValueError(f"dataset is missing table attribute(s): {missing}.")
    time_period = getattr(dataset, "time_period", None)
    if not hasattr(dataset, "household_weight_kind"):
        raise TypeError(
            "dataset.household_weight_kind is required on in-memory datasets; "
            "defaulting an absent kind would silently downgrade importance or "
            "calibrated weights to design. Declare the kind explicitly "
            "(H5 paths keep their documented attribute-less design default)."
        )
    weight_kind = dataset.household_weight_kind
    mass_log = getattr(dataset, "mass_log", ())
    if mass_log is None:
        raise TypeError(
            "dataset.mass_log must be a tuple of MassChangeRecord, not None; "
            "omit the attribute entirely for an empty history."
        )
    mass_log = tuple(mass_log)
    return {
        "person": dataset.person.copy(),
        "benunit": dataset.benunit.copy(),
        "household": dataset.household.copy(),
        "time_period": _normalise_time_period(time_period, source_year=source_year),
        "household_weight_kind": weight_kind,
        "mass_log": mass_log,
    }


def _read_uk_single_year_h5(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    if dataset_path.suffix != ".h5":
        raise ValueError("UK single-year dataset path must end with '.h5'.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"UK single-year dataset not found: {dataset_path}.")
    weight_kind, mass_log = read_uk_single_year_weight_metadata(dataset_path)
    with pd.HDFStore(dataset_path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(UK_SINGLE_YEAR_TABLES) - keys)
        if missing:
            raise ValueError(f"UK single-year dataset is missing table(s): {missing}.")
        time_period = str(store["time_period"].iloc[0])
        return {
            "person": store["person"],
            "benunit": store["benunit"],
            "household": store["household"],
            "time_period": time_period,
            "household_weight_kind": weight_kind,
            "mass_log": mass_log,
        }


def _validate_input_tables(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    household_id_column: str,
    household_weight_column: str,
) -> None:
    _require_columns(person, PERSON_ID_COLUMNS, label="person")
    _require_columns(benunit, BENUNIT_ID_COLUMNS, label="benunit")
    _require_columns(household, [household_id_column], label="household")
    _require_columns(household, [household_weight_column], label="household")
    _require_unique(person, "person_id", label="person")
    _require_unique(benunit, "benunit_id", label="benunit")
    _require_unique(household, household_id_column, label="household")


def _validate_weight_metadata(
    household_weight_kind: WeightKind,
    mass_log: tuple[MassChangeRecord, ...],
) -> None:
    if not isinstance(household_weight_kind, WeightKind):
        raise TypeError(
            "household_weight_kind must be a WeightKind, got "
            f"{household_weight_kind!r}."
        )
    if not isinstance(mass_log, tuple) or any(
        not isinstance(record, MassChangeRecord) for record in mass_log
    ):
        raise TypeError("mass_log must be a tuple of MassChangeRecord.")


def _assert_household_mass_conserved(
    input_total: float,
    output_total: float,
) -> None:
    """Require the clone's exact mass-conservation bound to hold."""

    if not (np.isfinite(input_total) and np.isfinite(output_total)):
        raise ValueError(
            "rowwise clone mass totals must be finite, got "
            f"{input_total!r} vs {output_total!r}."
        )
    if not np.isclose(
        output_total,
        input_total,
        rtol=MASS_CONSERVATION_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise ValueError(
            "rowwise clone leaked household mass: input total "
            f"{input_total!r} vs cloned total {output_total!r} exceeds the "
            f"declared relative tolerance {MASS_CONSERVATION_RELATIVE_TOLERANCE}."
        )


def apply_uk_source_lineage_modulus(
    household: pd.DataFrame,
    *,
    modulus: int,
    household_id_column: str = "household_id",
) -> pd.DataFrame:
    """Derive pool-grain lineage as ``pool_source_household_id``.

    The certified UK pool encodes its 10x clone tiers as
    ``household_id = tier * 10**8 + base``, so a modulus of ``10**8`` recovers
    the enhanced-FRS pool source. The derived column is a *distinct lineage
    layer*: any immediate-layer ``source_household_id``/``source_household_key``
    the input carries (the national staging H5 does) is left untouched. The
    modulus is only meaningful for rows whose ids follow the pool's tier
    scheme — on a seam output, channel-rebuilt households (e.g. the rebuilt
    SPI channel) carry ids outside that scheme, so pool-lineage diagnostics
    must be read per support channel.

    The mapping is refused when it would be ambiguous (the pool column
    already exists) or vacuous (no id reaches the modulus, making it the
    identity), and household ids must be finite non-negative integers —
    validated on the numeric values, not after a lossy integer cast.
    """

    if not isinstance(modulus, int) or isinstance(modulus, bool) or modulus <= 0:
        raise ValueError("source_lineage_modulus must be a positive integer.")
    if POOL_SOURCE_LINEAGE_COLUMN in household.columns:
        raise ValueError(
            f"household already carries {POOL_SOURCE_LINEAGE_COLUMN!r}; "
            "applying source_lineage_modulus would be ambiguous."
        )
    numeric = pd.to_numeric(household[household_id_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("source_lineage_modulus requires finite household ids.")
    if (numeric < 0).any():
        raise ValueError("source_lineage_modulus requires non-negative household ids.")
    if (numeric != np.floor(numeric)).any():
        raise ValueError("source_lineage_modulus requires integral household ids.")
    ids = (
        pd.to_numeric(household[household_id_column], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    if (ids // modulus == 0).all():
        raise ValueError(
            f"source_lineage_modulus={modulus} exceeds every household_id and "
            "would be an identity mapping; pass the pool's real clone-tier "
            "offset."
        )
    frame = household.copy()
    frame[POOL_SOURCE_LINEAGE_COLUMN] = ids % modulus
    return frame


def _assert_mass_log_current(
    mass_log: tuple[MassChangeRecord, ...],
    household_total: float,
) -> None:
    """Refuse a household mass log whose latest record disagrees with reality.

    Without this, a stale incoming chain would be hidden by the clone's own
    self-consistent appended record and pass downstream validation.
    """

    household_records = [record for record in mass_log if record.entity == "household"]
    if not household_records:
        return
    latest = household_records[-1]
    if not np.isclose(
        latest.new_total,
        household_total,
        rtol=MASS_CONSERVATION_RELATIVE_TOLERANCE,
        atol=0.0,
    ):
        raise ValueError(
            "household mass log is stale: latest household record new_total "
            f"{latest.new_total!r} disagrees with the actual household weight "
            f"total {household_total!r}; refusing to extend a broken chain."
        )


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} table is missing column(s): {missing}.")


def _require_unique(frame: pd.DataFrame, column: str, *, label: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{label}.{column} contains missing values.")
    if frame[column].duplicated().any():
        duplicates = frame.loc[frame[column].duplicated(), column].unique()
        raise ValueError(
            f"{label}.{column} must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )


def _normalise_time_period(
    value: int | str | None,
    *,
    source_year: int | None,
) -> str:
    if value is None:
        if source_year is None:
            raise ValueError("time_period or source_year is required.")
        value = source_year
    return str(value)
