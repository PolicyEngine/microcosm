"""Chunked loader for the 2024 ACS 1-year national PUMS archives.

The loader reads only the columns used by the ACS spine, joins housing and
person records through the Census ``SERIALNO`` key, and delegates US entity
construction to :func:`populace.frame.assign_us_unit_structure`, matching the
existing ASEC pool.  Source monetary columns remain native here; the separate
ACS input-mapping stage applies the Census adjustment factors and records
which PolicyEngine inputs are native versus transferred.

Full national archives contain multiple CSV members.  Each member is read in
bounded chunks, but the final selected-column tables necessarily materialize:
the returned :class:`~populace.frame.Frame` itself is the dense base-pool
artifact.  Vacant housing records are recorded and omitted because a Frame
cannot contain a household with no person membership.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from populace.frame import US_SCHEMA, Frame, WeightKind, Weights
from populace.frame.units import assign_us_unit_structure

__all__ = [
    "ACS_2024_1YR_SPINE",
    "AcsPumsSource",
    "build_acs_pums_unit_frame",
    "load_acs_pums_tables",
]

ACS_2024_1YR_SPINE = "acs_2024_1yr"
ACS_2024_1YR_VINTAGE = 2024
DEFAULT_CHUNKSIZE = 100_000

_HOUSEHOLD_REQUIRED = (
    "SERIALNO",
    "PUMA",
    "WGTP",
    "NP",
    "ADJHSG",
    "TEN",
    "RNTP",
    "GRNTP",
    "TAXAMT",
    "TYPEHUGQ",
)
_HOUSEHOLD_STATE_COLUMNS = ("ST", "STATE")
_HOUSEHOLD_FRAME_COLUMNS = (
    "NP",
    "ADJHSG",
    "TEN",
    "RNTP",
    "GRNTP",
    "TAXAMT",
    "TYPEHUGQ",
)
_PERSON_REQUIRED = (
    "SERIALNO",
    "SPORDER",
    "RELSHIPP",
    "AGEP",
    "SEX",
    "MAR",
    "ADJINC",
    "WAGP",
    "SEMP",
    "SSP",
    "SSIP",
    "RETP",
    "INTP",
    "PWGTP",
)
_PERSON_OPTIONAL: tuple[str, ...] = ()

# Temporary aliases consumed only by microunit's dependent gross-income test.
# ACS combined sources stay combined: INTP is placed on one gross-income
# component rather than split across interest/dividend/rent by invented shares.
_MICROUNIT_INCOME_ALIASES = {
    "WAGP": "WSAL_VAL",
    "SEMP": "SEMP_VAL",
    "INTP": "INT_VAL",
    "RETP": "PNSN_VAL",
    "SSP": "SS_VAL",
}

_ACS_REFERENCE_PERSON = 20
_ACS_SPOUSE_CODES = frozenset({21, 23})
_ACS_CHILD_CODES = frozenset({25, 26, 27})
_ACS_TO_CPS_MARITAL_STATUS = {
    1: 3,  # married, spouse absent until a RELSHIPP spouse is found
    2: 4,  # widowed
    3: 5,  # divorced
    4: 6,  # separated
    5: 7,  # never married
}
_ACS_RELATED_CODES = frozenset(
    {
        21,
        23,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        35,
    }
)

# ACS RELSHIPP -> microunit CPS relationship code. Reference persons and
# spouses are resolved separately because their CPS codes depend on household
# composition / sex. GQ persons are kept as nonrelatives without relatives.
_ACS_TO_CPS_RELATIONSHIP = {
    22: 13,  # opposite-sex unmarried partner
    24: 13,  # same-sex unmarried partner
    25: 5,  # biological child
    26: 5,  # adopted child
    27: 5,  # stepchild
    28: 9,  # sibling
    29: 8,  # parent
    30: 7,  # grandchild
    31: 10,  # parent-in-law
    32: 10,  # child-in-law
    33: 10,  # other relative
    34: 13,  # roommate / housemate
    35: 11,  # foster child
    36: 14,  # other nonrelative
    37: 14,  # institutional GQ person
    38: 14,  # noninstitutional GQ person
}


@dataclass(frozen=True)
class AcsPumsSource:
    """Local Census PUMS archives used to construct one ACS spine."""

    household_zip: Path
    person_zip: Path
    vintage: int = ACS_2024_1YR_VINTAGE
    max_households: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "household_zip", Path(self.household_zip))
        object.__setattr__(self, "person_zip", Path(self.person_zip))
        if self.vintage != ACS_2024_1YR_VINTAGE:
            raise ValueError(
                "ACS multispine v1 is pinned to the 2024 1-year PUMS; "
                f"got vintage {self.vintage}."
            )
        if self.max_households is not None and self.max_households <= 0:
            raise ValueError("max_households must be positive when provided.")


def load_acs_pums_tables(
    source: AcsPumsSource,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Read and key-align the household and person PUMS tables.

    Only selected native columns are materialized. Census blanks remain
    missing; this stage never converts an out-of-universe blank to zero.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    household, household_members = _read_archive(
        source.household_zip,
        member_prefix="psam_hus",
        required=_HOUSEHOLD_REQUIRED,
        optional=_HOUSEHOLD_STATE_COLUMNS,
        chunksize=chunksize,
    )
    state_column = next(
        (column for column in _HOUSEHOLD_STATE_COLUMNS if column in household),
        None,
    )
    if state_column is None:
        raise ValueError(
            "ACS household CSV member(s) missing required state column "
            "('ST' or 'STATE')."
        )
    if state_column != "ST":
        household = household.rename(columns={state_column: "ST"})

    duplicate_households = household["SERIALNO"].duplicated(keep=False)
    if duplicate_households.any():
        examples = household.loc[duplicate_households, "SERIALNO"].head().tolist()
        raise ValueError(f"ACS duplicate household SERIALNO value(s): {examples}.")
    household = household.sort_values("SERIALNO", kind="stable").reset_index(drop=True)
    all_household_serials = frozenset(household["SERIALNO"].tolist())
    household, vacant_count = _occupied_households(household)
    if source.max_households is not None and len(household) > source.max_households:
        household = _smoke_household_selection(household, source.max_households)
    selected_serials = frozenset(household["SERIALNO"].tolist())

    person, person_members = _read_archive(
        source.person_zip,
        member_prefix="psam_pus",
        required=_PERSON_REQUIRED,
        optional=_PERSON_OPTIONAL,
        chunksize=chunksize,
        valid_serials=all_household_serials,
        retained_serials=selected_serials,
    )
    duplicate_people = person.duplicated(["SERIALNO", "SPORDER"], keep=False)
    if duplicate_people.any():
        examples = (
            person.loc[duplicate_people, ["SERIALNO", "SPORDER"]]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"ACS duplicate person key(s): {examples}.")
    person = person.sort_values(["SERIALNO", "SPORDER"], kind="stable").reset_index(
        drop=True
    )

    if household.empty or person.empty:
        raise ValueError("ACS source selection contains no occupied household records.")
    _validate_person_counts(household, person)

    metadata: dict[str, Any] = {
        "spine": ACS_2024_1YR_SPINE,
        "vintage": source.vintage,
        "household_csv_members": household_members,
        "person_csv_members": person_members,
        "household_rows": int(len(household)),
        "person_rows": int(len(person)),
        "vacant_household_rows_dropped": vacant_count,
        "max_households": source.max_households,
    }
    return {"household": household, "person": person}, metadata


def build_acs_pums_unit_frame(
    source: AcsPumsSource,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[Frame, dict[str, Any]]:
    """Construct the ACS 2024 1-year US entity frame."""

    tables, metadata = load_acs_pums_tables(source, chunksize=chunksize)
    household = tables["household"].copy()
    person = tables["person"].copy()

    serial_to_id = pd.Series(
        np.arange(1, len(household) + 1, dtype=np.int64),
        index=household["SERIALNO"],
    )
    person["household_id"] = person["SERIALNO"].map(serial_to_id)
    if person["household_id"].isna().any():  # pragma: no cover - guarded above
        raise ValueError("ACS person-to-household id mapping produced missing ids.")
    person["household_id"] = person["household_id"].astype("int64")
    person = _with_structural_columns(person)
    person["source_year"] = source.vintage
    person["source_household_id"] = person["SERIALNO"].astype(str)
    person["source_person_id"] = _required_integer(person, "SPORDER")
    person["source_row_id"] = (
        ACS_2024_1YR_SPINE
        + ":"
        + person["SERIALNO"].astype(str)
        + ":"
        + person["SPORDER"].astype("int64").astype(str)
    )

    household_weights = _household_weights(household, person)
    # SERIALNO belongs on the household table in the returned Frame. Its
    # source identity remains available per person through source_* lineage.
    unit_input = _with_microunit_income_aliases(person.drop(columns=["SERIALNO"]))
    strata = pd.Series(
        ACS_2024_1YR_SPINE,
        index=unit_input.index,
        dtype="object",
        name="stratum",
    )
    frame = assign_us_unit_structure(
        unit_input,
        year=source.vintage,
        household_weights=household_weights,
        schema=US_SCHEMA,
        strata=strata,
    )
    frame = _attach_household_source_columns(frame, household)
    metadata.update(
        {
            "weighted_household_population": frame.weights_for("household").total,
            "relationship_pointer_policy": (
                "RELSHIPP reference/spouse pairing; own/adopted/stepchildren "
                "point to reference person and present spouse; all other "
                "parent pointers remain absent"
            ),
            "group_quarters_weight_policy": (
                "PWGTP for the one person represented by a WGTP=0 GQ "
                "placeholder; WGTP otherwise"
            ),
        }
    )
    return frame, metadata


def _read_archive(
    path: Path,
    *,
    member_prefix: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    chunksize: int,
    valid_serials: frozenset[str] | None = None,
    retained_serials: frozenset[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"ACS PUMS archive not found: {path}")
    pieces: list[pd.DataFrame] = []
    with ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.lower().startswith(member_prefix)
            and name.lower().endswith(".csv")
        )
        if not members:
            raise ValueError(
                f"ACS archive {path} has no {member_prefix}*.csv member(s)."
            )
        for member_name in members:
            with archive.open(member_name) as member:
                columns = pd.read_csv(member, nrows=0).columns.tolist()
            missing = sorted(set(required) - set(columns))
            if missing:
                raise ValueError(
                    f"ACS CSV member {member_name!r} missing required "
                    f"column(s): {missing}."
                )
            usecols = [column for column in (*required, *optional) if column in columns]
            string_columns = {
                column: "string"
                for column in ("SERIALNO", "ST", "STATE", "PUMA")
                if column in usecols
            }
            with archive.open(member_name) as member:
                reader = pd.read_csv(
                    member,
                    usecols=usecols,
                    dtype=string_columns,
                    chunksize=chunksize,
                    low_memory=False,
                )
                for chunk in reader:
                    if valid_serials is not None:
                        orphan = ~chunk["SERIALNO"].isin(valid_serials)
                        if orphan.any():
                            examples = (
                                chunk.loc[orphan, "SERIALNO"]
                                .drop_duplicates()
                                .head()
                                .tolist()
                            )
                            raise ValueError(
                                "ACS person SERIALNO value(s) missing from the "
                                f"household archive: {examples}."
                            )
                    if retained_serials is not None:
                        chunk = chunk.loc[chunk["SERIALNO"].isin(retained_serials)]
                    if not chunk.empty:
                        pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=[*required, *optional]), members
    return pd.concat(pieces, ignore_index=True), members


def _occupied_households(household: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    people = pd.to_numeric(household["NP"], errors="coerce")
    if people.isna().any() or (people < 0).any() or (people % 1 != 0).any():
        raise ValueError("ACS NP must be a finite nonnegative integer.")
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    if kind.isna().any() or (~kind.isin([1, 2, 3])).any():
        bad = sorted(kind.loc[kind.isna() | ~kind.isin([1, 2, 3])].unique().tolist())
        raise ValueError(f"ACS TYPEHUGQ contains unsupported code(s): {bad}.")
    weight = pd.to_numeric(household["WGTP"], errors="coerce")
    gq = kind.isin([2, 3])
    if (people.loc[gq] != 1).any():
        raise ValueError("ACS group-quarters placeholders must have NP=1.")
    if (weight.loc[gq] != 0).any() or (weight.loc[~gq] <= 0).any():
        raise ValueError(
            "ACS TYPEHUGQ/WGTP disagree: GQ weights must be 0 and housing-unit "
            "weights must be positive."
        )
    vacant = people == 0
    if (gq & vacant).any():  # pragma: no cover - implied by NP=1 guard
        raise ValueError("ACS group-quarters placeholders cannot be vacant.")
    return household.loc[~vacant].reset_index(drop=True), int(vacant.sum())


def _smoke_household_selection(
    household: pd.DataFrame,
    max_households: int,
) -> pd.DataFrame:
    """Choose a stable, geography-agnostic smoke subset, prioritizing HUs."""

    serials = household["SERIALNO"].astype(str).tolist()
    kinds = pd.to_numeric(household["TYPEHUGQ"], errors="raise").to_numpy()

    def rank(position: int) -> tuple[int, bytes]:
        housing_priority = 0 if kinds[position] == 1 else 1
        digest = hashlib.sha256(serials[position].encode()).digest()
        return housing_priority, digest

    selected = heapq.nsmallest(max_households, range(len(household)), key=rank)
    return household.iloc[sorted(selected)].reset_index(drop=True)


def _validate_person_counts(
    household: pd.DataFrame,
    person: pd.DataFrame,
) -> None:
    observed = person.groupby("SERIALNO", sort=False).size()
    expected = pd.Series(
        pd.to_numeric(household["NP"], errors="raise").to_numpy(dtype=np.int64),
        index=household["SERIALNO"],
    )
    aligned = observed.reindex(expected.index).fillna(0).astype(np.int64)
    mismatch = aligned != expected
    if mismatch.any():
        examples = [
            {
                "SERIALNO": str(serial),
                "NP": int(expected.loc[serial]),
                "person_rows": int(aligned.loc[serial]),
            }
            for serial in expected.index[mismatch][:5]
        ]
        raise ValueError(f"ACS NP/person row-count mismatch: {examples}.")


def _with_structural_columns(person: pd.DataFrame) -> pd.DataFrame:
    result = person.copy()
    result["PH_SEQ"] = result["household_id"].astype("int64")
    result["A_LINENO"] = _required_integer(result, "SPORDER")
    result["A_AGE"] = _required_integer(result, "AGEP")
    marital_status = _required_integer(result, "MAR")
    unknown_marital = sorted(set(marital_status) - set(_ACS_TO_CPS_MARITAL_STATUS))
    if unknown_marital:
        raise ValueError(f"ACS MAR contains unsupported code(s): {unknown_marital}.")
    result["A_MARITL"] = np.asarray(
        [_ACS_TO_CPS_MARITAL_STATUS[value] for value in marital_status],
        dtype=np.int64,
    )
    result["A_SPOUSE"] = np.zeros(len(result), dtype=np.int64)
    result["PEPAR1"] = np.zeros(len(result), dtype=np.int64)
    result["PEPAR2"] = np.zeros(len(result), dtype=np.int64)
    result["A_EXPRRP"] = np.zeros(len(result), dtype=np.int64)

    relationships = _required_integer(result, "RELSHIPP")
    sexes = _required_integer(result, "SEX")
    unknown = sorted(
        set(relationships)
        - {
            _ACS_REFERENCE_PERSON,
            *_ACS_SPOUSE_CODES,
            *_ACS_TO_CPS_RELATIONSHIP,
        }
    )
    if unknown:
        raise ValueError(f"ACS RELSHIPP contains unsupported code(s): {unknown}.")

    for _household_id, index in result.groupby(
        "household_id", sort=False
    ).groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        household_relationships = relationships[positions]
        reference_positions = positions[
            household_relationships == _ACS_REFERENCE_PERSON
        ]
        gq_only = np.isin(household_relationships, [37, 38]).all()
        if gq_only:
            result.loc[positions, "A_EXPRRP"] = 14
            continue
        if len(reference_positions) != 1:
            raise ValueError(
                "ACS housing-unit household must contain exactly one "
                f"RELSHIPP=20 reference person; household id {_household_id!r} "
                f"contains {len(reference_positions)}."
            )
        reference_position = int(reference_positions[0])
        reference_line = int(result.loc[reference_position, "A_LINENO"])
        spouse_positions = positions[
            np.isin(household_relationships, list(_ACS_SPOUSE_CODES))
        ]
        if len(spouse_positions) > 1:
            raise ValueError(
                "ACS household contains multiple spouse records for reference "
                f"person; household id {_household_id!r}."
            )
        spouse_line = 0
        if len(spouse_positions) == 1:
            spouse_position = int(spouse_positions[0])
            if (
                marital_status[reference_position] != 1
                or marital_status[spouse_position] != 1
            ):
                raise ValueError(
                    "ACS RELSHIPP spouse pair must have MAR=1 for both people; "
                    f"household id {_household_id!r}."
                )
            spouse_line = int(result.loc[spouse_position, "A_LINENO"])
            result.loc[reference_position, "A_SPOUSE"] = spouse_line
            result.loc[spouse_position, "A_SPOUSE"] = reference_line
            result.loc[[reference_position, spouse_position], "A_MARITL"] = 1

        has_relatives = bool(
            np.isin(household_relationships, list(_ACS_RELATED_CODES)).any()
        )
        result.loc[reference_position, "A_EXPRRP"] = 1 if has_relatives else 2
        for position in positions:
            relationship = int(relationships[position])
            if relationship == _ACS_REFERENCE_PERSON:
                continue
            if relationship in _ACS_SPOUSE_CODES:
                result.loc[position, "A_EXPRRP"] = 3 if sexes[position] == 1 else 4
                continue
            result.loc[position, "A_EXPRRP"] = _ACS_TO_CPS_RELATIONSHIP[relationship]
            if relationship in _ACS_CHILD_CODES:
                result.loc[position, "PEPAR1"] = reference_line
                result.loc[position, "PEPAR2"] = spouse_line
    return result


def _with_microunit_income_aliases(person: pd.DataFrame) -> pd.DataFrame:
    """Expose adjusted ACS income to tax-unit construction, then discard it.

    The aliases are not model inputs and are removed when household source
    columns are attached. They exist solely so microunit's qualifying-relative
    gross-income test sees the ACS amounts instead of treating every optional
    component as absent.
    """

    if "ADJINC" not in person:
        observed = [column for column in _MICROUNIT_INCOME_ALIASES if column in person]
        if observed:
            raise ValueError(
                f"ACS income column(s) {observed} require adjustment column 'ADJINC'."
            )
        return person
    result = person.copy()
    adjustment = pd.to_numeric(result["ADJINC"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    for source, alias in _MICROUNIT_INCOME_ALIASES.items():
        if source not in result:
            continue
        amount = pd.to_numeric(result[source], errors="coerce").to_numpy(
            dtype=np.float64
        )
        observed = ~np.isnan(amount)
        if (observed & ~np.isfinite(amount)).any():
            raise ValueError(f"ACS income source {source!r} must be finite.")
        invalid = observed & (~np.isfinite(adjustment) | (adjustment <= 0))
        if invalid.any():
            raise ValueError(
                "ACS ADJINC must be finite and positive wherever "
                f"{source!r} is observed."
            )
        result[alias] = amount * (adjustment / 1_000_000.0)
    return result


def _household_weights(household: pd.DataFrame, person: pd.DataFrame) -> Weights:
    raw = pd.to_numeric(household["WGTP"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("ACS WGTP must be finite and nonnegative.")
    gq = raw == 0
    if gq.any():
        if "PWGTP" not in person:
            raise ValueError("ACS WGTP=0 GQ records require person PWGTP.")
        person_counts = person.groupby("SERIALNO").size()
        gq_serials = household.loc[gq, "SERIALNO"]
        bad_counts = gq_serials[person_counts.reindex(gq_serials).to_numpy() != 1]
        if len(bad_counts):
            raise ValueError(
                "ACS WGTP=0 GQ placeholder must map to exactly one person; "
                f"bad SERIALNO value(s): {bad_counts.head().tolist()}."
            )
        gq_people = person.loc[person["SERIALNO"].isin(gq_serials)]
        person_weight = (
            gq_people.set_index("SERIALNO")["PWGTP"]
            .pipe(pd.to_numeric, errors="coerce")
            .reindex(gq_serials)
            .to_numpy(dtype=np.float64)
        )
        if not np.isfinite(person_weight).all() or (person_weight <= 0).any():
            raise ValueError("ACS GQ PWGTP must be finite and positive.")
        raw[gq] = person_weight
    return Weights(raw, WeightKind.DESIGN)


def _attach_household_source_columns(
    frame: Frame,
    source_household: pd.DataFrame,
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    aliases = [
        alias
        for alias in _MICROUNIT_INCOME_ALIASES.values()
        if alias in tables["person"]
    ]
    if aliases:
        tables["person"] = tables["person"].drop(columns=aliases)
    household = tables["household"]
    source = source_household.reset_index(drop=True)
    if len(household) != len(source):  # pragma: no cover - structural guard
        raise ValueError("ACS household source rows do not align to unit frame.")
    household["SERIALNO"] = source["SERIALNO"].astype(str).to_numpy()
    household["ST"] = source["ST"].astype(str).str.zfill(2).to_numpy()
    household["PUMA"] = source["PUMA"].astype(str).str.zfill(5).to_numpy()
    household["state_fips"] = pd.to_numeric(source["ST"], errors="raise").to_numpy(
        dtype=np.int64
    )
    household["puma_geoid"] = household["ST"] + household["PUMA"]
    # The PUMA ladder's canonical anchor is the full seven-digit
    # state_fips+PUMA5CE geoid. Keep the raw five-digit Census value in PUMA
    # for source lineage, and expose the ladder-ready geoid through both the
    # canonical ``puma`` column and its explicit alias.
    household["puma"] = household["puma_geoid"].to_numpy()
    for column in _HOUSEHOLD_FRAME_COLUMNS:
        household[column] = source[column].to_numpy()
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _required_integer(frame: pd.DataFrame, column: str) -> np.ndarray:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"ACS required structural column {column!r} contains blanks.")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(
            f"ACS required structural column {column!r} contains non-integers."
        )
    return values.astype(np.int64)
