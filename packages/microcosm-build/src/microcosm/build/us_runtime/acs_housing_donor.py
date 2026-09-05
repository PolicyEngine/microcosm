"""ACS2024 reported housing donor, independent of any recipient population.

This projection does not expose ``pre_subsidy_rent``. The 2024 ACS Subject
Definitions (p.16) instruct respondents to exclude housing-authority/agency
subsidies from contract rent. The separate ACS2022 donor and its nominal
graph remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
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


@dataclass(frozen=True)
class Acs2024HousingDonor:
    """One row per housing-unit reference person, keyed by source SERIALNO."""

    table: pd.DataFrame
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Acs2024ReportedRentPreparation:
    """Numeric QRF training for reported rent; no recipient/application stage."""

    training: pd.DataFrame
    predictors: tuple[str, ...]
    provenance: dict[str, Any]


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
    identities = {}
    manifest = load_acs_source_manifest()
    for role, path in (
        ("household", source.household_zip),
        ("person", source.person_zip),
    ):
        size, digest = _file_identity(path)
        if authenticated:
            _verify_download(manifest.artifact(role), size=size, digest=digest)
        identities[role] = {"sha256": digest, "size_bytes": size}
    household, h_members = _read_archive(
        source.household_zip,
        member_prefix="psam_hus",
        required=_HOUSEHOLD_COLUMNS,
        optional=("ST", "STATE"),
        chunksize=chunksize,
    )
    person, p_members = _read_archive(
        source.person_zip,
        member_prefix="psam_pus",
        required=_PERSON_COLUMNS,
        optional=(),
        chunksize=chunksize,
        valid_serials=frozenset(household["SERIALNO"]),
    )
    result = _project(household, person)
    result.provenance.update(
        {
            "source_evidence": "authenticated_archives"
            if authenticated
            else "synthetic_fixture_archives",
            "archives": identities,
            "csv_members": {"household": h_members, "person": p_members},
        }
    )
    return result


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


def _project(household: pd.DataFrame, person: pd.DataFrame) -> Acs2024HousingDonor:
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
    return Acs2024HousingDonor(
        table,
        {
            **json.loads(resource),
            "recipe_sha256": hashlib.sha256(resource).hexdigest(),
            "source_evidence": "synthetic_fixture_tables",
            "household_rows": len(table),
            "linked_person_rows": len(p),
            "vacant_rows_excluded": vacant_count,
            "gq_rows_excluded": int(gq.sum()),
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
    for flag in ("FRNTP", "FTAXP"):
        if not donor.table[flag].isin([0, 1]).all():
            raise ValueError(f"ACS prepared donor {flag} must contain 0/1.")
    if donor.table["source_household_id"].duplicated().any():
        raise ValueError("ACS prepared donor household IDs must be unique.")
    adapter = donor.table.rename(
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
        "price_year": 2024,
        "income_reference": "reference person's past-12-month amounts adjusted by ADJINC",
        "target": _TARGET,
        "target_universe": "occupied TEN=3, finite RNTP, FRNTP=0",
    }
    feature_schema_sha256 = hashlib.sha256(
        json.dumps(feature_schema, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Acs2024ReportedRentPreparation(
        training,
        predictors,
        {
            **donor.provenance,
            "target": _TARGET,
            "category_levels": levels,
            "sample_rows": len(sample),
            "fit_rows": len(training),
            "weight_kind": "design",
            "feature_schema": feature_schema,
            "feature_schema_sha256": feature_schema_sha256,
        },
    )
