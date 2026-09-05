"""ACS2024 reported housing donor, independent of any recipient population.

This projection does not expose ``pre_subsidy_rent``. The 2024 ACS Subject
Definitions (p.16) describe the respondent instruction to exclude housing-
authority/agency subsidies. This is publisher documentation, not an independently
authenticated 2024 respondent-help pamphlet. The separate ACS2022 donor and its nominal
graph remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.table_identity import native_table_identity
from microcosm.build.us_runtime.acs_inputs import _adjusted_dollars
from microcosm.build.us_runtime.acs_pums import (
    DEFAULT_CHUNKSIZE,
    AcsPumsSource,
    _occupied_households,
    _read_archive,
    _validate_person_counts,
)
from microcosm.build.us_runtime.acs_sources import (
    _file_identity,
    _verify_download,
    load_acs_source_manifest,
)
from microcosm.build.us_runtime.housing_inputs import _archived_joint_training_sample

__all__ = [
    "Acs2024HousingDonor",
    "Acs2024ReportedRentPreparation",
    "load_acs_2024_housing_donor",
    "load_synthetic_acs_2024_housing_donor",
    "project_synthetic_acs_2024_housing_donor",
    "prepare_acs_2024_reported_rent",
]

_HOUSEHOLD_COLUMNS = (
    "SERIALNO",
    "WGTP",
    "NP",
    "TYPEHUGQ",
    "TEN",
    "ADJHSG",
    "RNTP",
    "TAXAMT",
    "FRNTP",
    "FTAXP",
)
_PERSON_COLUMNS = (
    "SERIALNO",
    "SPORDER",
    "RELSHIPP",
    "AGEP",
    "SEX",
    "ADJINC",
    "WAGP",
    "SEMP",
    "SSP",
    "RETP",
)
_INCOMES = {
    "WAGP": "employment_income_before_lsr",
    "SEMP": "self_employment_income_before_lsr",
    "SSP": "acs_social_security_income",
    "RETP": "acs_retirement_income",
}
_NUMERIC_PREDICTORS = ("age", "is_male", *_INCOMES.values(), "household_size")
_CATEGORICAL_PREDICTORS = ("tenure_type", "state_code_str")
_TARGET = "acs_annual_reported_contract_rent"
_STATES = frozenset(
    {
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        53,
        54,
        55,
        56,
    }
)


@dataclass(frozen=True, init=False)
class Acs2024HousingDonor:
    """Producer-sealed projection; table changes are refused at preparation.

    Use a loader or the synthetic projector. Metadata access returns a detached
    document; caller dictionaries cannot mint archive-authenticated evidence.
    """

    table: pd.DataFrame
    _receipt_bytes: bytes = field(repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Use an ACS housing donor loader or synthetic projector.")

    @property
    def provenance(self) -> dict[str, Any]:
        """Return detached metadata from the immutable producer receipt."""
        return json.loads(self._receipt_bytes)


@dataclass(frozen=True, init=False)
class Acs2024ReportedRentPreparation:
    """Sealed training; consumers obtain a checked copy via verified_training."""

    training: pd.DataFrame
    predictors: tuple[str, ...]
    _receipt_bytes: bytes = field(repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("Use prepare_acs_2024_reported_rent to create training.")

    @property
    def provenance(self) -> dict[str, Any]:
        """Return detached metadata from the immutable derived receipt."""
        return json.loads(self._receipt_bytes)

    def verified_training(self) -> pd.DataFrame:
        """Verify native content and predictor order, returning an owned copy.

        Call at the fit boundary; direct access to ``training`` is inspectable
        mutable data and is not a verified consumption operation.
        """
        if type(self) is not Acs2024ReportedRentPreparation:
            raise TypeError("Training consumption requires an issued ACS preparation.")
        table, receipt = _verified_snapshot(
            self.training, self._receipt_bytes, "training_identity"
        )
        if tuple(receipt["feature_schema"]["ordered_predictors"]) != self.predictors:
            raise ValueError("ACS training predictor order differs from its receipt.")
        return table


def _receipt_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _verified_snapshot(
    table: pd.DataFrame, receipt: bytes, key: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Copy before verification, then consume this exact copy. Mutation after
    # verification cannot change the local table used by preparation or fitting.
    snapshot = table.copy(deep=True)
    # pandas deep-copies cells but shares the axes' underlying arrays. Detach
    # both before hashing so later public axis mutation cannot alter this copy.
    snapshot.index = table.index.copy(deep=True)
    snapshot.columns = table.columns.copy(deep=True)
    document = json.loads(receipt)
    try:
        identity = native_table_identity(snapshot)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "ACS table content identity is no longer supported."
        ) from error
    if identity != document[key]:
        raise ValueError(
            "ACS table content identity differs from its producer receipt."
        )
    return snapshot, document


def _seal_donor(table: pd.DataFrame, provenance: dict[str, Any]) -> Acs2024HousingDonor:
    result = object.__new__(Acs2024HousingDonor)
    object.__setattr__(result, "table", table)
    object.__setattr__(
        result,
        "_receipt_bytes",
        _receipt_bytes(
            provenance | {"projection_identity": native_table_identity(table)}
        ),
    )
    return result


def _seal_preparation(
    table: pd.DataFrame, predictors: tuple[str, ...], provenance: dict[str, Any]
) -> Acs2024ReportedRentPreparation:
    result = object.__new__(Acs2024ReportedRentPreparation)
    object.__setattr__(result, "training", table)
    object.__setattr__(result, "predictors", predictors)
    object.__setattr__(
        result,
        "_receipt_bytes",
        _receipt_bytes(
            provenance | {"training_identity": native_table_identity(table)}
        ),
    )
    return result


def load_acs_2024_housing_donor(
    source: AcsPumsSource, *, chunksize: int = DEFAULT_CHUNKSIZE
) -> Acs2024HousingDonor:
    """Verify the existing packaged archive pins before projecting raw data.

    This never downloads. A source subset is refused: donor selection cannot
    inherit a recipient's ``max_households`` limit.
    """
    return _load(source, chunksize=chunksize, authenticated=True)


def load_synthetic_acs_2024_housing_donor(
    source: AcsPumsSource, *, chunksize: int = DEFAULT_CHUNKSIZE
) -> Acs2024HousingDonor:
    """Exercise the same archive projection on explicitly synthetic fixtures."""
    return _load(source, chunksize=chunksize, authenticated=False)


def _load(
    source: AcsPumsSource, *, chunksize: int, authenticated: bool
) -> Acs2024HousingDonor:
    if source.vintage != 2024 or source.max_households is not None:
        raise ValueError("ACS housing donor requires the independent full 2024 source.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")
    manifest = load_acs_source_manifest()
    identities, snapshots = {}, {}
    # Private disk snapshots bound memory at archive scale. Hash the closed
    # snapshots and parse those same files, never reopen the caller's path.
    # Directory ownership/read-only files prevent ordinary concurrent cache
    # writers from changing the verified input; this is not hostile-process isolation.
    with TemporaryDirectory(prefix="microcosm-acs-housing-") as directory:
        for role, path in (
            ("household", source.household_zip),
            ("person", source.person_zip),
        ):
            snapshot = Path(directory) / f"{role}.zip"
            with path.open("rb") as original, snapshot.open("xb") as destination:
                while chunk := original.read(1024 * 1024):
                    destination.write(chunk)
            snapshot.chmod(0o400)
            size, digest = _file_identity(snapshot)
            if authenticated:
                _verify_download(manifest.artifact(role), size=size, digest=digest)
            snapshots[role] = snapshot
            identities[role] = {"sha256": digest, "size_bytes": size}
        household, h_members = _read_archive(
            snapshots["household"],
            member_prefix="psam_hus",
            required=_HOUSEHOLD_COLUMNS,
            optional=("ST", "STATE"),
            chunksize=chunksize,
        )
        person, p_members = _read_archive(
            snapshots["person"],
            member_prefix="psam_pus",
            required=_PERSON_COLUMNS,
            optional=(),
            chunksize=chunksize,
            valid_serials=frozenset(household["SERIALNO"]),
        )
        return _project(
            household,
            person,
            source_provenance={
                "source_evidence": "authenticated_archives"
                if authenticated
                else "synthetic_fixture_archives",
                "archives": identities,
                "csv_members": {"household": h_members, "person": p_members},
            },
        )


def project_synthetic_acs_2024_housing_donor(
    household: pd.DataFrame, person: pd.DataFrame
) -> Acs2024HousingDonor:
    """Project invented native tables; never claim authenticated provenance."""
    return _project(household, person)


def _numeric(table: pd.DataFrame, column: str, *, nullable: bool = False) -> pd.Series:
    values = pd.to_numeric(table[column], errors="raise")
    if (
        (values.notna() & ~np.isfinite(values)) | (values.isna() & (not nullable))
    ).any():
        raise ValueError(f"ACS {column} contains missing/nonfinite values.")
    return values


def _project(
    household: pd.DataFrame,
    person: pd.DataFrame,
    *,
    source_provenance: dict[str, Any] | None = None,
) -> Acs2024HousingDonor:
    household, person = household.copy(), person.copy()
    for table, columns in ((household, _HOUSEHOLD_COLUMNS), (person, _PERSON_COLUMNS)):
        missing = set(columns) - set(table)
        if missing:
            raise ValueError(
                f"ACS housing donor missing source columns: {sorted(missing)}."
            )
        serial = table["SERIALNO"]
        if (
            serial.isna().any()
            or not serial.map(lambda x: isinstance(x, str) and bool(x.strip())).all()
        ):
            raise ValueError("ACS SERIALNO must contain nonblank string IDs.")
        table["SERIALNO"] = serial.astype(CANONICAL_STRING_DTYPE)
    if household["SERIALNO"].duplicated().any():
        raise ValueError("ACS household SERIALNO must be unique.")
    if (~person["SERIALNO"].isin(household["SERIALNO"])).any():
        raise ValueError("ACS person references missing household SERIALNO.")
    person["SPORDER"] = _numeric(person, "SPORDER")
    if (~person["SPORDER"].between(1, 20) | (person["SPORDER"] % 1 != 0)).any():
        raise ValueError("ACS SPORDER must be an integer from 1 to 20.")
    if person.duplicated(["SERIALNO", "SPORDER"]).any():
        raise ValueError("ACS person SERIALNO/SPORDER must be unique.")
    for column in ("NP", "WGTP", "TYPEHUGQ"):
        household[column] = _numeric(household, column)
    if (
        (household["WGTP"] % 1 != 0).any()
        or (household["WGTP"] > 9999).any()
        or (household["NP"] > 20).any()
    ):
        raise ValueError("ACS WGTP must be integral and at most 9999; NP at most 20.")
    occupied, vacant_count = _occupied_households(household)
    _validate_person_counts(household, person)
    gq = household["TYPEHUGQ"].isin([2, 3])
    for flag in ("FRNTP", "FTAXP"):
        values = _numeric(household, flag, nullable=True)
        if (~values.loc[~gq].isin([0, 1])).any() or values.loc[gq].notna().any():
            raise ValueError(
                f"ACS {flag} requires 0/1 for housing units and blank for GQ."
            )
        household[flag] = values
    # Exclusion is an explicit source-universe choice, not zero-weight GQ training.
    selected = occupied.loc[occupied["TYPEHUGQ"] == 1, "SERIALNO"]
    h = (
        household.loc[household["SERIALNO"].isin(selected)]
        .set_index("SERIALNO")
        .sort_index()
    )
    if h.empty:
        raise ValueError("ACS housing donor has no occupied housing units.")
    p = person.loc[person["SERIALNO"].isin(h.index)].copy()
    relationship = _numeric(p, "RELSHIPP")
    if (~relationship.isin(range(20, 37))).any():
        raise ValueError("ACS housing-unit RELSHIPP must be a code from 20 to 36.")
    heads = p.loc[relationship == 20].set_index("SERIALNO")
    if heads.index.duplicated().any() or set(heads.index) != set(h.index):
        raise ValueError(
            "ACS housing unit requires exactly one RELSHIPP=20 reference person."
        )
    heads = heads.reindex(h.index)
    tenure = _numeric(h, "TEN")
    if (~tenure.isin([1, 2, 3, 4])).any():
        raise ValueError("ACS occupied TEN must be one of 1, 2, 3, 4.")
    state_columns = [c for c in ("ST", "STATE") if c in h]
    if not state_columns:
        raise ValueError("ACS housing donor requires ST or STATE.")
    state = _numeric(h, state_columns[0])
    if len(state_columns) == 2 and not state.eq(_numeric(h, state_columns[1])).all():
        raise ValueError("ACS ST and STATE disagree.")
    if (~state.isin(_STATES)).any():
        raise ValueError("ACS housing donor contains a non-US/DC state code.")
    age, sex = _numeric(heads, "AGEP"), _numeric(heads, "SEX")
    if (~age.between(0, 99) | (age % 1 != 0)).any() or (~sex.isin([1, 2])).any():
        raise ValueError(
            "ACS reference-person AGEP/SEX is outside the codebook domain."
        )
    table = pd.DataFrame(
        {
            "source_household_id": h.index,
            "source_head_person_number": heads["SPORDER"].to_numpy(dtype=np.int64),
            "age": age,
            "is_male": (sex == 1).astype(float),
            "TEN": tenure,
            "tenure_type": tenure.map(
                {
                    1: "OWNED_WITH_MORTGAGE",
                    2: "OWNED_WITH_MORTGAGE",
                    3: "RENTED",
                    4: "NONE",
                }
            ),
            "state_code_str": state.map(lambda x: f"{int(x):02d}"),
            "household_size": p.groupby("SERIALNO").size().reindex(h.index),
            "household_weight": h["WGTP"].astype(float),
            "FRNTP": h["FRNTP"].astype(int),
            "FTAXP": h["FTAXP"].astype(int),
        },
        index=h.index,
    )
    for raw, output in _INCOMES.items():
        amounts = _numeric(heads, raw)
        if raw != "SEMP" and (amounts < 0).any():
            raise ValueError(f"ACS {raw} must be nonnegative.")
        table[output] = _adjusted_dollars(amounts, heads, factor="ADJINC")
    for raw, output, universe, scale, lower, upper in (
        ("RNTP", _TARGET, tenure == 3, 12, 1, 99999),
        ("TAXAMT", "acs_annual_property_tax", tenure.isin([1, 2]), 1, 0, 57500),
    ):
        amounts = _numeric(h, raw, nullable=True)
        if amounts.loc[~universe].notna().any() or amounts.loc[universe].isna().any():
            raise ValueError(f"ACS {raw} disagrees with its occupied TEN universe.")
        if (
            ~amounts.loc[universe].between(lower, upper)
            | amounts.loc[universe].mod(1).ne(0)
        ).any():
            raise ValueError(f"ACS {raw} is outside the codebook dollar range.")
        table[output] = _adjusted_dollars(amounts, h, factor="ADJHSG") * scale
    table["rent_observed"] = table[_TARGET].notna() & table["FRNTP"].eq(0)
    table["property_tax_observed"] = table["acs_annual_property_tax"].notna() & table[
        "FTAXP"
    ].eq(0)
    resource = Path(__file__).with_name("acs_2024_housing_donor.json").read_bytes()
    return _seal_donor(
        table,
        {
            **json.loads(resource),
            "recipe_sha256": hashlib.sha256(resource).hexdigest(),
            "source_evidence": "synthetic_fixture_tables",
            "household_rows": len(table),
            "linked_person_rows": len(p),
            "vacant_rows_excluded": vacant_count,
            "gq_rows_excluded": int(gq.sum()),
            **(source_provenance or {}),
        },
    )


def prepare_acs_2024_reported_rent(
    donor: Acs2024HousingDonor,
) -> Acs2024ReportedRentPreparation:
    """Reuse the joint sampler, fitting only unallocated reported rent.

    Out-of-universe rent remains missing. Unlike the legacy donor's filled
    zeros, this target is defined only for renters; constant tenure therefore
    needs no dummy column. Encoded columns describe donors only, and are not
    an API for mapping any recipient to pre-subsidy rent.
    """
    if type(donor) is not Acs2024HousingDonor:
        raise TypeError("Preparation requires an issued ACS housing donor.")
    table, provenance = _verified_snapshot(
        donor.table, donor._receipt_bytes, "projection_identity"
    )
    for flag in ("FRNTP", "FTAXP"):
        if not table[flag].isin([0, 1]).all():
            raise ValueError(f"ACS prepared donor {flag} must contain 0/1.")
    if table["source_household_id"].duplicated().any():
        raise ValueError("ACS prepared donor household IDs must be unique.")
    adapter = table.rename(
        columns={
            _TARGET: "rent",
            "acs_annual_property_tax": "real_estate_taxes",
        }
    ).copy()
    adapter["rent_is_allocated"] = adapter["FRNTP"].eq(1)
    adapter["real_estate_taxes_is_allocated"] = adapter["FTAXP"].eq(1)
    sample, masks = _archived_joint_training_sample(adapter)
    selected = sample.loc[masks["rent"]].set_index("source_household_id")
    training = selected.loc[:, _NUMERIC_PREDICTORS].copy()
    levels = {}
    for column in _CATEGORICAL_PREDICTORS:
        levels[column] = sorted(selected[column].unique().tolist())
        for level in levels[column][1:]:
            training[f"{column}__{level}"] = selected[column].eq(level).astype(float)
    predictors = tuple(training.columns)
    training[_TARGET] = selected["rent"]
    training["household_weight"] = selected["household_weight"]
    if not np.isfinite(training.to_numpy(dtype=float)).all():
        raise ValueError("ACS reported-rent training must be finite.")
    if (training[_TARGET] <= 0).any():
        raise ValueError("ACS reported contract rent must be positive where observed.")
    if (training["household_weight"] <= 0).any():
        raise ValueError("ACS reported-rent training requires positive design weights.")
    feature_schema = {
        "schema": "microcosm.us.acs2024_reported_contract_rent_features.v1",
        "ordered_predictors": list(predictors),
        "numeric_predictors": list(_NUMERIC_PREDICTORS),
        "category_levels": levels,
        "categorical_encoding": "sorted levels, omit first; constant feature has no dummy",
        "category_level_source": "rent fit rows after archived joint target sample",
        "unknown_category_policy": "refuse",
        "price_year": 2024,
        "income_reference": "reference person's past-12-month amounts adjusted by ADJINC",
        "target": _TARGET,
        "target_universe": "occupied TEN=3, finite RNTP, FRNTP=0",
    }
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(feature_schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return _seal_preparation(
        training,
        predictors,
        {
            **provenance,
            "donor_receipt_sha256": hashlib.sha256(donor._receipt_bytes).hexdigest(),
            "target": _TARGET,
            "category_levels": levels,
            "sample_rows": len(sample),
            "fit_rows": len(training),
            "weight_kind": "design",
            "feature_schema": feature_schema,
            "feature_schema_sha256": feature_schema_sha256,
        },
    )
