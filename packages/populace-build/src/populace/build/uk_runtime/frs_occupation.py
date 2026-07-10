"""Observed FRS occupation (SOC 2020 major group) onto the UK person table.

Provenance
----------
The UKDA Family Resources Survey (UK Data Service, **End User Licence**)
ships an ``adult.tab`` file with one row per adult respondent, keyed by
``SERNUM`` (household serial number), ``BENUNIT`` (benefit unit within the
household) and ``PERSON`` (person within the benefit unit). Its ``SOC2020``
variable carries the respondent's Standard Occupational Classification 2020
at **major-group** level, coded as thousands: ``1000`` (managers, directors
and senior officials) through ``9000`` (elementary occupations). Like all
UKDS EUL microdata (see :mod:`populace.build.uk_runtime.spi_support` and the
DONOR CONTRACT in :mod:`populace.build.uk_runtime.exposure_imputation`), the
file lives outside the repository and its path is supplied by the caller —
the raw microdata are **never committed**; tests exercise the contract with
synthetic tables only.

Join keys
---------
PolicyEngine-UK single-year person IDs are derived from the same FRS keys:
``person_id = SERNUM * 100 + BENUNIT * 10 + PERSON`` (with ``benunit_id =
SERNUM * 10 + BENUNIT`` and ``household_id = SERNUM``). This module rebuilds
that composite ID from ``adult.tab``'s key columns and joins it against the
person table's ``person_id`` (the first of
:data:`populace.build.uk_runtime.rowwise_dataset.PERSON_ID_COLUMNS`), the
same single-year ID surface :mod:`~populace.build.uk_runtime.spi_support`
clones from.

Missingness contract
--------------------
* Children have no ``adult.tab`` row, so they receive NaN.
* Adults whose ``SOC2020`` is missing or invalid (anything outside the
  ``1000``-``9000`` major-group coding, e.g. FRS sentinel ``-1`` or ``0``)
  keep NaN; the report counts them so a build log shows the gap.

Contract with exposure imputation
---------------------------------
The produced person column is
:data:`~populace.build.uk_runtime.exposure_imputation.SOC_MAJOR_GROUP_COLUMN`
(``soc_major_group``), kept in the **raw ``1000``-``9000`` coding**:
:mod:`~populace.build.uk_runtime.exposure_imputation` documents that any
consistent numeric coding works as long as donor and target match, and
:func:`populace.build.uk_runtime.ai_exposure.exposure_for_major_group`
accepts the coding via :func:`frs_major_group_to_digit` (or directly after
that normalisation). :data:`UK_FRS_OCCUPATION_STAGE_NAME` produces the
column, so in a :class:`~populace.build.plan.StagePlan` this stage slots
immediately before
:func:`~populace.build.uk_runtime.exposure_imputation.exposure_imputation_stage`,
whose primary path consumes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.plan import DonorSpec, Stage
from populace.build.uk_runtime.exposure_imputation import SOC_MAJOR_GROUP_COLUMN
from populace.frame import Frame

__all__ = [
    "FRS_ADULT_KEY_COLUMNS",
    "FRS_ADULT_SOC_COLUMN",
    "FRS_OCCUPATION_DONOR",
    "SOC_MAJOR_GROUP_COLUMN",
    "UK_FRS_OCCUPATION_STAGE_NAME",
    "VALID_SOC_MAJOR_GROUPS",
    "attach_soc_major_group",
    "frs_major_group_to_digit",
    "frs_occupation_stage",
    "frs_person_ids",
    "load_frs_adult_table",
    "soc_major_group_for_persons",
]

#: The ``adult.tab`` columns identifying one adult respondent.
FRS_ADULT_KEY_COLUMNS = ("SERNUM", "BENUNIT", "PERSON")

#: The ``adult.tab`` SOC 2020 major-group variable (coded 1000-9000).
FRS_ADULT_SOC_COLUMN = "SOC2020"

#: The valid FRS major-group codes: thousands 1000 (managers) to 9000
#: (elementary occupations). Anything else (sentinels like -1, 0, or
#: missing) is treated as missing SOC.
VALID_SOC_MAJOR_GROUPS = tuple(range(1000, 10_000, 1000))

#: Stage name for the FRS occupation merge step of a UK build plan.
UK_FRS_OCCUPATION_STAGE_NAME = "frs_occupation"

#: The donor declaration a UK build plan records for this stage.
FRS_OCCUPATION_DONOR = DonorSpec(
    survey="DWP Family Resources Survey (UKDA, End User Licence)",
    source="UK Data Service, Family Resources Survey adult.tab (SOC2020 "
    "major-group variable, keyed SERNUM/BENUNIT/PERSON)",
    notes="UKDS EUL microdata, held locally and never committed. This is an "
    "observed merge, not an imputation: the adult's own 1-digit SOC 2020 "
    "major group is joined onto the PolicyEngine-UK single-year person IDs. "
    "Children and adults with missing/invalid SOC2020 stay NaN.",
)

_MISSING_SOC_WARN_KEY = "adults_missing_soc"


def load_frs_adult_table(
    path: str | Path,
    *,
    columns: tuple[str, ...] = (*FRS_ADULT_KEY_COLUMNS, FRS_ADULT_SOC_COLUMN),
) -> pd.DataFrame:
    """Read the UKDA FRS ``adult.tab`` (tab-delimited), keeping ``columns``.

    The path points at locally held UKDS End User Licence microdata; it is
    supplied by the caller and never committed.

    Raises:
        ValueError: If the file lacks any of ``columns``.
    """

    table = pd.read_csv(path, sep="\t", usecols=lambda name: name in set(columns))
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"FRS adult table at {path} is missing column(s): {missing}.")
    return table


def frs_person_ids(adult: pd.DataFrame) -> np.ndarray:
    """PolicyEngine-UK single-year person IDs from FRS adult key columns.

    ``person_id = SERNUM * 100 + BENUNIT * 10 + PERSON`` — the composite the
    PolicyEngine-UK FRS build assigns, and the ``person_id`` the single-year
    person table carries (see PERSON_ID_COLUMNS in
    :mod:`populace.build.uk_runtime.rowwise_dataset`).

    Raises:
        ValueError: On missing key columns or missing/non-numeric key values.
    """

    missing = sorted(set(FRS_ADULT_KEY_COLUMNS) - set(adult.columns))
    if missing:
        raise ValueError(f"adult table is missing key column(s): {missing}.")
    keys = {}
    for column in FRS_ADULT_KEY_COLUMNS:
        values = pd.to_numeric(adult[column], errors="raise")
        if values.isna().any():
            raise ValueError(f"adult.{column} contains missing values.")
        keys[column] = values.astype("int64").to_numpy()
    return keys["SERNUM"] * 100 + keys["BENUNIT"] * 10 + keys["PERSON"]


def soc_major_group_for_persons(
    person: pd.DataFrame,
    adult: pd.DataFrame,
    *,
    person_id_column: str = "person_id",
) -> tuple[pd.Series, dict[str, int]]:
    """Person-level ``soc_major_group`` series joined from ``adult.tab``.

    Joins the adult table onto ``person[person_id_column]`` via
    :func:`frs_person_ids` and keeps the raw ``1000``-``9000`` coding
    (:mod:`~populace.build.uk_runtime.exposure_imputation` documents that
    coding, and :func:`~populace.build.uk_runtime.ai_exposure.exposure_for_major_group`
    accepts it after :func:`frs_major_group_to_digit`).

    Returns:
        ``(series, report)`` — a float series named
        :data:`SOC_MAJOR_GROUP_COLUMN`, index-aligned to ``person``, NaN for
        persons without an adult row (children) or with missing/invalid
        ``SOC2020``; and a count report with keys ``n_persons``,
        ``adults_matched``, ``adults_with_valid_soc``,
        ``adults_missing_soc`` and ``persons_without_adult_row``.

    Raises:
        ValueError: On missing columns or duplicated adult person IDs.
    """

    if person_id_column not in person.columns:
        raise ValueError(f"person table is missing {person_id_column!r}.")
    if FRS_ADULT_SOC_COLUMN not in adult.columns:
        raise ValueError(f"adult table is missing {FRS_ADULT_SOC_COLUMN!r}.")

    adult_ids = frs_person_ids(adult)
    if pd.Index(adult_ids).has_duplicates:
        duplicates = pd.Index(adult_ids)
        duplicates = duplicates[duplicates.duplicated()].unique()
        raise ValueError(
            "adult table has duplicated SERNUM/BENUNIT/PERSON keys; duplicate "
            f"person ID(s): {list(map(str, duplicates[:5]))}."
        )

    soc = pd.to_numeric(adult[FRS_ADULT_SOC_COLUMN], errors="coerce").astype(float)
    soc = soc.where(soc.isin(VALID_SOC_MAJOR_GROUPS))
    lookup = pd.Series(soc.to_numpy(), index=adult_ids)

    person_ids = pd.to_numeric(person[person_id_column], errors="raise").astype("int64")
    values = person_ids.map(lookup)
    values.name = SOC_MAJOR_GROUP_COLUMN

    matched = person_ids.isin(lookup.index)
    report = {
        "n_persons": int(len(person)),
        "adults_matched": int(matched.sum()),
        "adults_with_valid_soc": int(values.notna().sum()),
        _MISSING_SOC_WARN_KEY: int((matched & values.isna()).sum()),
        "persons_without_adult_row": int((~matched).sum()),
    }
    return values, report


def frs_major_group_to_digit(codes: Any) -> np.ndarray:
    """Normalise FRS ``1000``-``9000`` major-group codes to ``"1"``-``"9"``.

    The 1-digit strings are what
    :func:`populace.build.uk_runtime.ai_exposure.exposure_for_major_group`
    indexes its major-group table by. Invalid/missing codes become ``None``.
    """

    values = pd.to_numeric(pd.Series(np.asarray(codes)), errors="coerce")
    digits = values.where(values.isin(VALID_SOC_MAJOR_GROUPS)) / 1000
    return np.array(
        [None if pd.isna(value) else str(int(value)) for value in digits],
        dtype=object,
    )


def attach_soc_major_group(
    frame: Frame,
    adult: pd.DataFrame,
    *,
    column: str = SOC_MAJOR_GROUP_COLUMN,
) -> Frame:
    """Return a new frame whose person table carries ``soc_major_group``.

    Mirrors :func:`~populace.build.uk_runtime.exposure_imputation.attach_exposure`'s
    reassembly, but NaN is *allowed* here — it is the documented value for
    children and for adults with missing/invalid SOC2020.

    Raises:
        ValueError: If ``column`` already exists on any entity table (the
            stage must run exactly once).
    """

    try:
        owner = frame.column_entity(column)
    except ValueError:
        owner = None
    if owner is not None:
        raise ValueError(
            f"Column {column!r} already exists on the {owner!r} table; "
            "attach_soc_major_group refuses to overwrite it. The FRS "
            "occupation stage should run exactly once."
        )

    values, _report = soc_major_group_for_persons(frame.person, adult)
    person_entity = frame.schema.person_entity
    person = frame.person.copy()
    person[column] = values.to_numpy(dtype=float)
    tables: dict[str, pd.DataFrame] = {person_entity: person}
    for entity in frame.schema.group_entities:
        tables[entity] = frame.table(entity)
    for link in frame.links:
        tables[link] = frame.link(link)
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    return Frame(
        tables,
        frame.schema,
        weights,
        frame.strata,
        mass_log=frame.mass_log,
    )


def frs_occupation_stage(
    adult: pd.DataFrame | str | Path,
    *,
    column: str = SOC_MAJOR_GROUP_COLUMN,
    donor: DonorSpec = FRS_OCCUPATION_DONOR,
) -> Stage:
    """Declare the FRS occupation-merge step of a UK build plan.

    Modeled on the :mod:`~populace.build.uk_runtime.spi_support` /
    :func:`~populace.build.uk_runtime.exposure_imputation.exposure_imputation_stage`
    pattern: the licensed local artifact (the ``adult.tab`` path or a
    pre-loaded frame) is closed over, and the stage, when run, joins the
    observed major group onto the frame's person table. It produces
    ``(column,)`` so it slots before ``exposure_imputation_stage`` — whose
    primary path consumes :data:`SOC_MAJOR_GROUP_COLUMN` — in a
    :class:`~populace.build.plan.StagePlan`.

    Args:
        adult: The FRS ``adult.tab`` path (read lazily inside the transform,
            so a bad path aborts the build loudly at run time) or an
            already-loaded adult DataFrame.
        column: The produced person column. Defaults to
            :data:`SOC_MAJOR_GROUP_COLUMN`.
        donor: The donor declaration recorded on the stage.

    Returns:
        A :class:`populace.build.plan.Stage` producing ``column`` and
        consuming ``person_id`` (the single-year ID the join keys on).
    """

    def transform(frame: Frame) -> Frame:
        table = (
            adult if isinstance(adult, pd.DataFrame) else load_frs_adult_table(adult)
        )
        return attach_soc_major_group(frame, table, column=column)

    return Stage(
        name=UK_FRS_OCCUPATION_STAGE_NAME,
        transform=transform,
        produces=(column,),
        consumes=("person_id",),
        donor=donor,
    )
