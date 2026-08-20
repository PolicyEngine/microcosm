"""CPS/ACS housing and tenure inputs carried by the retired eCPS build.

The archived pipeline used two primary sources for this family:

* CPS ASEC household ``H_TENURE`` mapped directly to household
  ``tenure_type``.  ASEC SPM ``SPM_CAPHOUSESUB`` and
  ``SPM_TENMORTSTATUS`` mapped directly to ``receives_housing_assistance``
  and ``spm_unit_tenure_type``.  Measured assistance receipt also anchors
  ``takes_up_housing_assistance_if_eligible`` without adding unobserved
  recipients; and
* the processed 2022 ACS public-use artifact supplied a 10-predictor rent
  donor.  The archived QRF selected household heads, built a shared 10,000-row
  sample for rent and real-estate taxes with target-specific allocation masks,
  and placed annual ``pre_subsidy_rent`` on the CPS household head.

Microcosm preserves those source semantics and uses its shared design-weighted
QRF.  Weighting the ACS fit is the sole deliberate strengthening of the
retired implementation: the archived fit loaded ``household_weight`` but did
not pass it through to microimpute.  No housing values are manufactured from
defaults or from the retired eCPS output.
"""

from __future__ import annotations

import hashlib
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    has_support_role_metadata,
    support_role_series,
)
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_2022_RENT_ARTIFACT_SHA256",
    "HOUSING_INPUTS_ARCHIVED_ACS_DERIVATION_URL",
    "HOUSING_INPUTS_ARCHIVED_CPS_RENT_URL",
    "HOUSING_INPUTS_ARCHIVED_CPS_SPM_URL",
    "HOUSING_INPUTS_ARCHIVED_PUF_IMPUTATION_URL",
    "HOUSING_TAKE_UP_ARCHIVED_DERIVATION_URL",
    "HOUSING_TAKE_UP_ARCHIVED_HUD_ETL_URL",
    "HOUSING_TAKE_UP_ARCHIVED_PARAMETER_URL",
    "US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS",
    "US_HOUSING_INPUTS_OUTPUT_COLUMNS",
    "US_HOUSING_INPUTS_STAGE_NAME",
    "US_HOUSING_ASSISTANCE_PUF_MAX_TRAIN_SAMPLES",
    "US_HOUSING_ASSISTANCE_PUF_N_ESTIMATORS",
    "US_HOUSING_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_HOUSING_NONCONSTANT_PERSON_COLUMNS",
    "US_HOUSING_NONCONSTANT_SPM_UNIT_COLUMNS",
    "US_HOUSING_PERSON_OUTPUT_COLUMNS",
    "US_HOUSING_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS",
    "US_HOUSING_REQUIRED_PERSON_SOURCE_COLUMNS",
    "US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS",
    "derive_us_housing_inputs",
    "impute_us_pre_subsidy_rent",
    "impute_us_housing_assistance_to_puf_support",
    "load_acs_2022_rent_donor",
    "us_housing_inputs_signal_gate",
    "us_housing_inputs_stage_spec",
    "us_housing_inputs_summary",
    "with_us_housing_inputs",
]

_ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
_ARCHIVED_ROOT = (
    "https://github.com/PolicyEngine/"
    + "policyengine-"
    + "us-data/blob/"
    + _ARCHIVED_COMMIT
)
HOUSING_INPUTS_ARCHIVED_CPS_RENT_URL = (
    _ARCHIVED_ROOT + "/policyengine_" + "us_data/datasets/cps/cps.py#L417-L535"
)
HOUSING_INPUTS_ARCHIVED_CPS_SPM_URL = (
    _ARCHIVED_ROOT + "/policyengine_" + "us_data/datasets/cps/cps.py#L1612-L1635"
)
HOUSING_INPUTS_ARCHIVED_ACS_DERIVATION_URL = (
    _ARCHIVED_ROOT + "/policyengine_" + "us_data/datasets/acs/acs.py#L82-L135"
)
HOUSING_INPUTS_ARCHIVED_PUF_IMPUTATION_URL = (
    _ARCHIVED_ROOT + "/policyengine_" + "us_data/datasets/cps/extended_cps.py#L135-L194"
)
HOUSING_TAKE_UP_ARCHIVED_DERIVATION_URL = (
    _ARCHIVED_ROOT + "/policyengine_" + "us_data/datasets/cps/cps.py#L664-L682"
)
HOUSING_TAKE_UP_ARCHIVED_PARAMETER_URL = (
    _ARCHIVED_ROOT
    + "/policyengine_"
    + "us_data/parameters/take_up/housing_assistance.yaml#L1-L15"
)
HOUSING_TAKE_UP_ARCHIVED_HUD_ETL_URL = (
    _ARCHIVED_ROOT + "/policyengine_" + "us_data/db/etl_housing_assistance.py#L35-L169"
)

US_HOUSING_INPUTS_STAGE_NAME = "acs_rent"
US_HOUSING_PERSON_OUTPUT_COLUMNS: tuple[str, ...] = ("pre_subsidy_rent",)
US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS: tuple[str, ...] = ("tenure_type",)
US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS: tuple[str, ...] = (
    "receives_housing_assistance",
    "takes_up_housing_assistance_if_eligible",
    "spm_unit_tenure_type",
)
US_HOUSING_INPUTS_OUTPUT_COLUMNS: tuple[str, ...] = (
    *US_HOUSING_PERSON_OUTPUT_COLUMNS,
    *US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS,
    *US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS,
)
US_HOUSING_NONCONSTANT_PERSON_COLUMNS = US_HOUSING_PERSON_OUTPUT_COLUMNS
US_HOUSING_NONCONSTANT_SPM_UNIT_COLUMNS = US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS
US_HOUSING_NONCONSTANT_HOUSEHOLD_COLUMNS = US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS

US_HOUSING_REQUIRED_PERSON_SOURCE_COLUMNS: tuple[str, ...] = (
    "SPM_CAPHOUSESUB",
    "SPM_TENMORTSTATUS",
    "person_spm_unit_id",
    "person_household_id",
    "is_household_head",
    "age",
    "is_female",
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
)
US_HOUSING_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS: tuple[str, ...] = (
    "H_TENURE",
    "state_fips",
)

# SHA-256 of the hermetic processed ACS_2022 ARRAYS artifact used by Build J.
# The loader below does not trust the local generating checkout: it validates
# the exact arrays and entity relationships documented by the immutable
# archived implementation before exposing a donor.
ACS_2022_RENT_ARTIFACT_SHA256 = (
    "0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4"
)

ACS_RENT_PREDICTORS: tuple[str, ...] = (
    "is_household_head",
    "age",
    "is_male",
    "tenure_type",
    "employment_income",
    "self_employment_income",
    "social_security",
    "pension_income",
    "state_code_str",
    "household_size",
)
_DONOR_WEIGHT_COLUMN = "household_weight"
_DONOR_ALLOCATION_COLUMN = "rent_is_allocated"
_DONOR_REAL_ESTATE_TAX_COLUMN = "real_estate_taxes"
_DONOR_REAL_ESTATE_TAX_ALLOCATION_COLUMN = "real_estate_taxes_is_allocated"
_MAX_TRAIN_SAMPLES = 10_000
_DEFAULT_N_ESTIMATORS = 100
US_HOUSING_ASSISTANCE_PUF_N_ESTIMATORS = 100
US_HOUSING_ASSISTANCE_PUF_MAX_TRAIN_SAMPLES = 5_000
_RENT_SHARE_BAND = (0.05, 0.25)
_HOUSING_ASSISTANCE_SHARE_BAND = (0.005, 0.08)

_HOUSEHOLD_TENURE_MAP = {
    0: "NONE",
    1: "OWNED_WITH_MORTGAGE",
    2: "RENTED",
    3: "NONE",
}
_HOUSEHOLD_TENURE_CODES = {
    "NONE": 0.0,
    "OWNED_WITH_MORTGAGE": 1.0,
    "OWNED_OUTRIGHT": 1.0,
    "RENTED": 2.0,
}
_ACS_CATEGORICAL_PREDICTORS = ("tenure_type", "state_code_str")
_SPM_TENURE_MAP = {
    1: "OWNER_WITH_MORTGAGE",
    2: "OWNER_WITHOUT_MORTGAGE",
    3: "RENTER",
}
_HOUSEHOLD_TENURE_VALUES = frozenset(_HOUSEHOLD_TENURE_CODES)
_SPM_TENURE_VALUES = frozenset(_SPM_TENURE_MAP.values())
_EXPECTED_HOUSEHOLD_TENURE_VALUES = frozenset({"NONE", "OWNED_WITH_MORTGAGE", "RENTED"})
_EXPECTED_SPM_TENURE_VALUES = frozenset(
    {"OWNER_WITH_MORTGAGE", "OWNER_WITHOUT_MORTGAGE", "RENTER"}
)

QRF: Any | None = None


def us_housing_inputs_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged ACS/CPS housing-stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_HOUSING_INPUTS_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_HOUSING_INPUTS_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_HOUSING_INPUTS_STAGE_NAME]
    missing = sorted(set(US_HOUSING_INPUTS_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_HOUSING_INPUTS_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_strings(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "S":
        return np.char.decode(raw, "utf-8")
    return raw.astype(str)


def _numeric_array(values: Any, *, name: str) -> np.ndarray:
    result = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(result).all():
        raise ValueError(f"Housing source column {name!r} contains nonfinite values.")
    return result


def load_acs_2022_rent_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = ACS_2022_RENT_ARTIFACT_SHA256,
    source_stream: BinaryIO | None = None,
) -> pd.DataFrame:
    """Load the archived processed ACS 2022 household-head rent donor.

    The loader reads entity arrays directly, aligns household variables through
    ``person_household_id``, collapses ACS owned-outright tenure exactly as the
    retired rent stage did, and retains only household heads.  Both archived
    target-allocation flags remain attached so the exact joint target sample
    can be replayed before the rent-specific fit.
    """

    import h5py

    source = Path(path)
    if source_stream is None and not source.exists():
        raise FileNotFoundError(f"ACS 2022 rent donor not found: {source}")
    if source_stream is None and expected_sha256 is not None:
        actual = _sha256(source)
        if actual != expected_sha256:
            raise ValueError(
                f"ACS 2022 rent donor SHA-256 mismatch: {actual} != {expected_sha256}."
            )

    person_columns = (
        "person_id",
        "person_household_id",
        "is_household_head",
        "age",
        "is_male",
        "employment_income",
        "self_employment_income",
        "social_security",
        "taxable_private_pension_income",
        "rent",
        "rent_is_allocated",
        "real_estate_taxes",
        "real_estate_taxes_is_allocated",
    )
    household_columns = (
        "household_id",
        "household_weight",
        "state_fips",
        "tenure_type",
    )
    h5_source: Path | BinaryIO
    if source_stream is None:
        h5_source = source
    else:
        source_stream.seek(0)
        h5_source = source_stream
    with h5py.File(h5_source, mode="r") as h5:
        missing = [
            column
            for column in (*person_columns, *household_columns)
            if column not in h5
        ]
        if missing:
            raise ValueError(f"ACS 2022 rent donor missing array(s): {missing}.")
        arrays = {
            column: np.asarray(h5[column])
            for column in (*person_columns, *household_columns)
        }

    person_n = len(arrays["person_id"])
    household_n = len(arrays["household_id"])
    bad_person_lengths = {
        column: len(arrays[column])
        for column in person_columns
        if len(arrays[column]) != person_n
    }
    bad_household_lengths = {
        column: len(arrays[column])
        for column in household_columns
        if len(arrays[column]) != household_n
    }
    if bad_person_lengths or bad_household_lengths:
        raise ValueError(
            "ACS 2022 rent donor entity-array lengths disagree: "
            f"person={bad_person_lengths}, household={bad_household_lengths}."
        )

    household_ids = np.asarray(arrays["household_id"])
    if pd.Index(household_ids).duplicated().any():
        raise ValueError("ACS 2022 household_id must be unique.")
    person_household_ids = np.asarray(arrays["person_household_id"])
    household_index = pd.Index(household_ids)
    positions = household_index.get_indexer(person_household_ids)
    if (positions < 0).any():
        bad = np.unique(person_household_ids[positions < 0])[:5].tolist()
        raise ValueError(
            f"ACS 2022 people reference missing household_id value(s): {bad}."
        )

    head_mask = np.asarray(arrays["is_household_head"], dtype=bool)
    head_household_ids = person_household_ids[head_mask]
    if pd.Index(head_household_ids).duplicated().any():
        raise ValueError("ACS 2022 rent donor has multiple heads in a household.")
    household_size = pd.Series(person_household_ids).value_counts(sort=False)

    household_tenure = _decode_strings(arrays["tenure_type"])
    normalized_tenure = np.asarray(
        [
            "OWNED_WITH_MORTGAGE" if value == "OWNED_OUTRIGHT" else value
            for value in household_tenure
        ],
        dtype=object,
    )
    unknown_tenure = sorted(set(normalized_tenure) - _HOUSEHOLD_TENURE_VALUES)
    if unknown_tenure:
        raise ValueError(
            f"ACS 2022 rent donor has unknown tenure value(s): {unknown_tenure}."
        )

    head_positions = positions[head_mask]
    donor = pd.DataFrame(
        {
            "is_household_head": np.ones(int(head_mask.sum()), dtype=np.float64),
            "age": _numeric_array(arrays["age"][head_mask], name="age"),
            "is_male": np.asarray(arrays["is_male"][head_mask], dtype=np.float64),
            "tenure_type": normalized_tenure[head_positions].astype(str),
            "employment_income": _numeric_array(
                arrays["employment_income"][head_mask], name="employment_income"
            ),
            "self_employment_income": _numeric_array(
                arrays["self_employment_income"][head_mask],
                name="self_employment_income",
            ),
            "social_security": _numeric_array(
                arrays["social_security"][head_mask], name="social_security"
            ),
            "pension_income": _numeric_array(
                arrays["taxable_private_pension_income"][head_mask],
                name="taxable_private_pension_income",
            ),
            "state_code_str": np.asarray(
                [f"{int(value):02d}" for value in arrays["state_fips"][head_positions]],
                dtype=object,
            ),
            "household_size": household_size.reindex(head_household_ids).to_numpy(
                dtype=np.float64
            ),
            "rent": _numeric_array(arrays["rent"][head_mask], name="rent"),
            _DONOR_ALLOCATION_COLUMN: np.asarray(
                arrays["rent_is_allocated"][head_mask], dtype=bool
            ),
            _DONOR_REAL_ESTATE_TAX_COLUMN: _numeric_array(
                arrays["real_estate_taxes"][head_mask], name="real_estate_taxes"
            ),
            _DONOR_REAL_ESTATE_TAX_ALLOCATION_COLUMN: np.asarray(
                arrays["real_estate_taxes_is_allocated"][head_mask], dtype=bool
            ),
            _DONOR_WEIGHT_COLUMN: _numeric_array(
                arrays["household_weight"][head_positions], name="household_weight"
            ),
        }
    )
    if (donor["rent"] < 0.0).any():
        raise ValueError("ACS 2022 rent donor contains negative rent values.")
    if (donor[_DONOR_REAL_ESTATE_TAX_COLUMN] < 0.0).any():
        raise ValueError(
            "ACS 2022 rent donor contains negative real-estate-tax values."
        )
    # Keep zero-WGTP group-quarters heads through the archived joint-target
    # sampler.  The retired unweighted fit retained them; Microcosm's deliberate
    # design-weighting strengthening gives them zero modeling mass without
    # changing which deterministic 10,000-row sample was selected.
    if (donor[_DONOR_WEIGHT_COLUMN] < 0.0).any():
        raise ValueError("ACS 2022 rent donor contains negative weights.")
    if float(donor[_DONOR_WEIGHT_COLUMN].sum()) <= 0.0:
        raise ValueError("ACS 2022 rent donor has no positive household weight.")
    if not (~donor[_DONOR_ALLOCATION_COLUMN]).any():
        raise ValueError("ACS 2022 rent donor has no unallocated rent observations.")
    if not (~donor[_DONOR_REAL_ESTATE_TAX_ALLOCATION_COLUMN]).any():
        raise ValueError(
            "ACS 2022 rent donor has no unallocated real-estate-tax observations."
        )
    return donor


def _required_columns(
    table: pd.DataFrame, columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"US housing inputs require {label} column(s): {missing}.")


def _constant_source_by_unit(
    person: pd.DataFrame,
    *,
    membership_column: str,
    source_column: str,
    unit_ids: pd.Series,
) -> np.ndarray:
    source = pd.to_numeric(person[source_column], errors="coerce")
    if source.isna().any():
        raise ValueError(
            f"US housing source {source_column!r} contains missing values."
        )
    replicas = pd.DataFrame(
        {
            "unit_id": person[membership_column].to_numpy(),
            "value": source.to_numpy(dtype=np.float64),
        }
    )
    bounds = replicas.groupby("unit_id", sort=False)["value"].agg(["min", "max"])
    unequal = bounds["min"].to_numpy() != bounds["max"].to_numpy()
    if unequal.any():
        bad = bounds.index.to_numpy()[unequal][:5].tolist()
        raise ValueError(
            f"US housing source {source_column!r} must be constant within its "
            f"SPM unit; unit id(s) {bad} disagree."
        )
    values = bounds["min"].reindex(unit_ids)
    if values.isna().any():
        bad = unit_ids.loc[values.isna()].head().tolist()
        raise ValueError(
            f"US housing source {source_column!r} does not cover SPM unit id(s) {bad}."
        )
    return values.to_numpy(dtype=np.float64)


def derive_us_housing_inputs(frame: Frame) -> Frame:
    """Carry the three exact ASEC housing/tenure inputs onto their entities."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US housing inputs require the US schema.")
    person = frame.table("person")
    household = frame.table("household")
    spm_unit = frame.table("spm_unit")
    _required_columns(
        person,
        ("SPM_CAPHOUSESUB", "SPM_TENMORTSTATUS", "person_spm_unit_id"),
        "person",
    )
    _required_columns(household, ("H_TENURE",), "household")

    raw_household_tenure = pd.to_numeric(
        household["H_TENURE"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(raw_household_tenure).all():
        raise ValueError("US housing source H_TENURE contains nonfinite values.")
    raw_household_codes = raw_household_tenure.astype(np.int64)
    if not np.array_equal(raw_household_tenure, raw_household_codes):
        raise ValueError("US housing source H_TENURE contains non-integer codes.")
    unknown_household_codes = sorted(
        set(raw_household_codes) - set(_HOUSEHOLD_TENURE_MAP)
    )
    if unknown_household_codes:
        raise ValueError(
            "US housing source H_TENURE contains unknown code(s): "
            f"{unknown_household_codes}."
        )

    subsidy = _constant_source_by_unit(
        person,
        membership_column="person_spm_unit_id",
        source_column="SPM_CAPHOUSESUB",
        unit_ids=spm_unit["spm_unit_id"],
    )
    if (subsidy < 0.0).any():
        raise ValueError("US housing source SPM_CAPHOUSESUB contains negative values.")
    raw_spm_tenure = _constant_source_by_unit(
        person,
        membership_column="person_spm_unit_id",
        source_column="SPM_TENMORTSTATUS",
        unit_ids=spm_unit["spm_unit_id"],
    )
    raw_spm_codes = raw_spm_tenure.astype(np.int64)
    if not np.array_equal(raw_spm_tenure, raw_spm_codes):
        raise ValueError("US housing source SPM_TENMORTSTATUS has non-integer codes.")
    unknown_spm_codes = sorted(set(raw_spm_codes) - set(_SPM_TENURE_MAP))
    if unknown_spm_codes:
        raise ValueError(
            "US housing source SPM_TENMORTSTATUS contains unknown code(s): "
            f"{unknown_spm_codes}."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"]["tenure_type"] = np.asarray(
        [_HOUSEHOLD_TENURE_MAP[code] for code in raw_household_codes], dtype=object
    )
    tables["spm_unit"]["receives_housing_assistance"] = subsidy > 0.0
    tables["spm_unit"]["takes_up_housing_assistance_if_eligible"] = subsidy > 0.0
    tables["spm_unit"]["spm_unit_tenure_type"] = np.asarray(
        [_SPM_TENURE_MAP[code] for code in raw_spm_codes], dtype=object
    )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _person_numeric(person: pd.DataFrame, *columns: str) -> np.ndarray:
    for column in columns:
        if column in person.columns:
            values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"US housing recipient column {column!r} contains nonfinite values."
                )
            return values
    raise ValueError(
        f"US housing recipient is missing every alternative column: {list(columns)}."
    )


def _person_sum(person: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    present = [column for column in columns if column in person.columns]
    if not present:
        raise ValueError(
            f"US housing recipient is missing component column(s): {list(columns)}."
        )
    values = np.zeros(len(person), dtype=np.float64)
    for column in present:
        values += _person_numeric(person, column)
    return values


def _strict_boolean_signal(
    values: pd.Series,
) -> tuple[np.ndarray, int, int]:
    """Normalize boolean/0/1 values while retaining missing/invalid counts."""

    raw = values.to_numpy(dtype=object)
    normalized = np.zeros(len(raw), dtype=bool)
    missing = 0
    invalid = 0
    for index, value in enumerate(raw):
        if pd.isna(value):
            missing += 1
            continue
        if isinstance(value, (bool, np.bool_)):
            normalized[index] = bool(value)
            continue
        if isinstance(value, (int, np.integer, float, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric) and numeric in (0.0, 1.0):
                normalized[index] = numeric == 1.0
                continue
        invalid += 1
    return normalized, missing, invalid


def _recipient_head_features(frame: Frame) -> tuple[pd.DataFrame, np.ndarray]:
    person = frame.table("person")
    household = frame.table("household")
    _required_columns(
        person,
        (
            "person_household_id",
            "is_household_head",
            "age",
            "is_female",
            "employment_income_before_lsr",
            "self_employment_income_before_lsr",
        ),
        "person",
    )
    _required_columns(
        household,
        ("household_id", "state_fips", "tenure_type"),
        "household",
    )

    head_mask = person["is_household_head"].fillna(False).astype(bool).to_numpy()
    heads = person.loc[head_mask]
    head_ids = heads["person_household_id"].to_numpy()
    if pd.Index(head_ids).duplicated().any():
        raise ValueError("US housing recipient selected multiple heads per household.")
    household_ids = household["household_id"].to_numpy()
    missing = sorted(set(household_ids) - set(head_ids))
    extra = sorted(set(head_ids) - set(household_ids))
    if missing or extra:
        raise ValueError(
            "US housing household-head alignment failed: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )

    tenure = household["tenure_type"].astype(str)
    unknown = sorted(set(tenure) - _HOUSEHOLD_TENURE_VALUES)
    if unknown:
        raise ValueError(
            f"US housing recipient has unknown tenure value(s): {unknown}."
        )
    tenure_by_id = pd.Series(tenure.to_numpy(), index=household_ids)
    state_by_id = pd.Series(
        pd.to_numeric(household["state_fips"], errors="coerce").to_numpy(),
        index=household_ids,
    )
    size_by_id = person["person_household_id"].value_counts(sort=False)

    social_security = _person_sum(
        person,
        (
            "social_security_retirement",
            "social_security_disability",
            "social_security_survivors",
            "social_security_dependents",
        ),
    )
    pension_income = _person_sum(
        person,
        (
            "taxable_private_pension_income",
            "tax_exempt_private_pension_income",
            "taxable_public_pension_income",
            "tax_exempt_public_pension_income",
        ),
    )
    all_features = pd.DataFrame(
        {
            "is_household_head": np.ones(len(person), dtype=np.float64),
            "age": _person_numeric(person, "age", "A_AGE"),
            "is_male": (~person["is_female"].fillna(False).astype(bool)).to_numpy(
                dtype=np.float64
            ),
            "employment_income": _person_numeric(
                person, "employment_income_before_lsr"
            ),
            "self_employment_income": _person_numeric(
                person, "self_employment_income_before_lsr"
            ),
            "social_security": social_security,
            "pension_income": pension_income,
        },
        index=person.index,
    ).loc[head_mask]
    all_features["tenure_type"] = pd.Series(head_ids).map(tenure_by_id).to_numpy()
    all_features["state_code_str"] = np.asarray(
        [f"{int(value):02d}" for value in pd.Series(head_ids).map(state_by_id)],
        dtype=object,
    )
    all_features["household_size"] = pd.Series(head_ids).map(size_by_id).to_numpy()
    all_features.index = head_ids
    aligned = all_features.reindex(household_ids)
    for column in set(ACS_RENT_PREDICTORS) - set(_ACS_CATEGORICAL_PREDICTORS):
        numeric = pd.to_numeric(aligned[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(numeric).all():
            raise ValueError(
                f"US housing recipient predictor {column!r} contains nonfinite values."
            )
        aligned[column] = numeric
    return aligned.loc[:, list(ACS_RENT_PREDICTORS)], head_mask


def _encode_acs_predictors(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Dummy-encode the retired QRF's string predictors without ordinality."""

    numeric_predictors = tuple(
        predictor
        for predictor in ACS_RENT_PREDICTORS
        if predictor not in _ACS_CATEGORICAL_PREDICTORS
    )
    encoded_training = training.loc[:, numeric_predictors].copy()
    encoded_prediction = prediction.loc[:, numeric_predictors].copy()
    encoded_columns = list(numeric_predictors)
    for column in _ACS_CATEGORICAL_PREDICTORS:
        training_values = training[column].astype(str)
        prediction_values = prediction[column].astype(str)
        levels = tuple(sorted(training_values.unique()))
        unknown = sorted(set(prediction_values.unique()) - set(levels))
        if unknown:
            raise ValueError(
                f"ACS rent recipient {column!r} has donor-unsupported value(s): "
                f"{unknown}."
            )
        if len(levels) < 2:
            raise ValueError(
                f"ACS rent donor categorical predictor {column!r} is constant."
            )
        for level in levels[1:]:
            dummy = f"{column}__{level}"
            encoded_training[dummy] = training_values.eq(level).to_numpy(
                dtype=np.float64
            )
            encoded_prediction[dummy] = prediction_values.eq(level).to_numpy(
                dtype=np.float64
            )
            encoded_columns.append(dummy)
    return encoded_training, encoded_prediction, tuple(encoded_columns)


def _stable_string_hash(value: str) -> np.uint64:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow encountered", RuntimeWarning)
        result = np.uint64(0)
        for byte in value.encode("utf-8"):
            result = result * np.uint64(31) + np.uint64(byte)
        result = result ^ (result >> np.uint64(33))
        result = result * np.uint64(0xFF51AFD7ED558CCD)
        result = result ^ (result >> np.uint64(33))
    return result


def _archived_training_rng(*, salt: str | None = None) -> np.random.Generator:
    key = "legacy_acs_rent_training_sample"
    if salt is not None:
        key = f"{key}:{salt}"
    return np.random.default_rng(int(_stable_string_hash(key)) % (2**63))


def _archived_joint_training_sample(
    donor: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Replay the retired rent/real-estate-tax target-filtered sample cap."""

    target_masks = {
        "rent": (
            np.isfinite(donor["rent"].to_numpy(dtype=np.float64))
            & ~donor[_DONOR_ALLOCATION_COLUMN].astype(bool).to_numpy()
        ),
        _DONOR_REAL_ESTATE_TAX_COLUMN: (
            np.isfinite(donor[_DONOR_REAL_ESTATE_TAX_COLUMN].to_numpy(dtype=np.float64))
            & ~donor[_DONOR_REAL_ESTATE_TAX_ALLOCATION_COLUMN].astype(bool).to_numpy()
        ),
    }
    for target, mask in target_masks.items():
        if not mask.any():
            raise ValueError(f"ACS rent donor has no observed rows for {target}.")

    union_mask = np.logical_or.reduce(tuple(target_masks.values()))
    union_positions = np.flatnonzero(union_mask)
    if len(union_positions) <= _MAX_TRAIN_SAMPLES:
        sample_positions = union_positions
    else:
        selected: list[int] = []
        selected_set: set[int] = set()
        per_target_cap = _MAX_TRAIN_SAMPLES // len(target_masks)
        for target, mask in target_masks.items():
            target_positions = np.flatnonzero(mask)
            target_sample = _archived_training_rng(salt=target).choice(
                target_positions,
                size=min(per_target_cap, len(target_positions)),
                replace=False,
            )
            for position in target_sample:
                integer_position = int(position)
                if integer_position not in selected_set:
                    selected.append(integer_position)
                    selected_set.add(integer_position)

        remaining_n = _MAX_TRAIN_SAMPLES - len(selected)
        if remaining_n > 0:
            remaining_positions = np.asarray(
                [
                    position
                    for position in union_positions
                    if int(position) not in selected_set
                ],
                dtype=np.int64,
            )
            if len(remaining_positions):
                fill_sample = _archived_training_rng(salt="fill").choice(
                    remaining_positions,
                    size=min(remaining_n, len(remaining_positions)),
                    replace=False,
                )
                selected.extend(int(position) for position in fill_sample)
        sample_positions = np.asarray(selected, dtype=np.int64)

    sampled = donor.iloc[sample_positions].copy().reset_index(drop=True)
    sampled_masks = {
        target: mask[sample_positions] for target, mask in target_masks.items()
    }
    return sampled, sampled_masks


def impute_us_pre_subsidy_rent(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> np.ndarray:
    """Draw annual ACS rent once per CPS household and place it on the head."""

    required = (
        *ACS_RENT_PREDICTORS,
        "rent",
        _DONOR_ALLOCATION_COLUMN,
        _DONOR_REAL_ESTATE_TAX_COLUMN,
        _DONOR_REAL_ESTATE_TAX_ALLOCATION_COLUMN,
        _DONOR_WEIGHT_COLUMN,
    )
    missing = [column for column in required if column not in donor.columns]
    if missing:
        raise ValueError(f"ACS rent donor table missing column(s): {missing}.")
    training_sample, target_masks = _archived_joint_training_sample(
        donor.loc[:, required].copy()
    )
    fit_frame = training_sample.loc[target_masks["rent"]].copy()
    if fit_frame.empty:
        raise ValueError("ACS rent donor has no observed training rows after sampling.")
    for column in (
        *(
            predictor
            for predictor in ACS_RENT_PREDICTORS
            if predictor not in _ACS_CATEGORICAL_PREDICTORS
        ),
        "rent",
        _DONOR_WEIGHT_COLUMN,
    ):
        fit_frame[column] = pd.to_numeric(fit_frame[column], errors="coerce")
        if not np.isfinite(fit_frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"ACS rent donor column {column!r} contains nonfinite values."
            )
    if (fit_frame["rent"] < 0.0).any():
        raise ValueError("ACS rent donor contains negative rent values.")
    if (fit_frame[_DONOR_WEIGHT_COLUMN] < 0.0).any():
        raise ValueError("ACS rent donor weights must be nonnegative.")
    if float(fit_frame[_DONOR_WEIGHT_COLUMN].sum()) <= 0.0:
        raise ValueError("ACS rent donor sampled weights sum to zero.")

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("microcosm.fit").QRF
    features, head_mask = _recipient_head_features(frame)
    encoded_training, encoded_features, encoded_predictors = _encode_acs_predictors(
        fit_frame,
        features,
    )
    encoded_training["rent"] = fit_frame["rent"].to_numpy(dtype=np.float64)
    encoded_training[_DONOR_WEIGHT_COLUMN] = fit_frame[_DONOR_WEIGHT_COLUMN].to_numpy(
        dtype=np.float64
    )
    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        encoded_training,
        predictors=list(encoded_predictors),
        targets=["rent"],
        weights=_DONOR_WEIGHT_COLUMN,
    )
    predicted = pd.to_numeric(
        fitted.predict(encoded_features)["rent"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(predicted).all():
        raise ValueError("ACS rent QRF produced nonfinite predictions.")
    predicted = np.maximum(predicted, 0.0)
    person_rent = np.zeros(frame.n("person"), dtype=np.float64)
    person_rent[head_mask] = predicted
    return person_rent


def _person_puf_predictors(frame: Frame) -> pd.DataFrame:
    """Build the retired eight CPS-to-PUF receiver predictors on people."""

    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    predictors = pd.DataFrame(index=person.index)
    predictors["age"] = _person_numeric(person, "age", "A_AGE")
    if "is_male" in person:
        predictors["is_male"] = person["is_male"].fillna(False).astype(bool)
    elif "is_female" in person:
        predictors["is_male"] = ~person["is_female"].fillna(False).astype(bool)
    elif "A_SEX" in person:
        predictors["is_male"] = _person_numeric(person, "A_SEX") == 1
    else:
        raise ValueError(
            "US housing PUF imputation requires is_male, is_female, or A_SEX."
        )
    if "has_esi" not in person:
        raise ValueError("US housing PUF imputation requires person.has_esi.")
    predictors["has_esi"] = person["has_esi"].fillna(False).astype(bool)

    filing_status_column = next(
        (
            column
            for column in ("filing_status_input", "filing_status")
            if column in tax_unit
        ),
        None,
    )
    if filing_status_column is None:
        raise ValueError("US housing PUF imputation requires tax-unit filing status.")
    filing_status = tax_unit[filing_status_column].map(
        lambda value: (
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        )
    )
    joint_by_id = pd.Series(
        filing_status.eq("JOINT").to_numpy(),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )
    predictors["tax_unit_is_joint"] = (
        person["person_tax_unit_id"].map(joint_by_id).fillna(False).astype(bool)
    )
    if "tax_unit_role_input" not in person:
        raise ValueError(
            "US housing PUF imputation requires tax_unit_role_input to count "
            "dependents."
        )
    dependent = (
        person["tax_unit_role_input"]
        .map(
            lambda value: (
                value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            )
        )
        .eq("DEPENDENT")
    )
    predictors["tax_unit_count_dependents"] = (
        dependent.groupby(person["person_tax_unit_id"])
        .transform("sum")
        .to_numpy(dtype=np.float64)
    )
    predictors["employment_income"] = _person_numeric(
        person, "employment_income_before_lsr", "WSAL_VAL"
    )
    predictors["self_employment_income"] = _person_numeric(
        person, "self_employment_income_before_lsr", "SEMP_VAL"
    )
    social_security_columns = (
        "social_security_retirement",
        "social_security_disability",
        "social_security_dependents",
        "social_security_survivors",
    )
    if all(column in person for column in social_security_columns):
        predictors["social_security"] = _person_sum(person, social_security_columns)
    else:
        predictors["social_security"] = _person_numeric(person, "SS_VAL")
    return predictors


def impute_us_housing_assistance_to_puf_support(
    frame: Frame,
    *,
    seed: int,
    n_estimators: int = US_HOUSING_ASSISTANCE_PUF_N_ESTIMATORS,
    max_train_samples: int = US_HOUSING_ASSISTANCE_PUF_MAX_TRAIN_SAMPLES,
) -> Frame:
    """Replace only the PUF clone's housing-assistance receipt flag by QRF.

    The archived extended-CPS stage included this one housing leaf in its
    common eight-predictor CPS-to-PUF fit, then reduced person predictions to
    the SPM unit by first-person value.  The measured ASEC half remains exact;
    rent and both tenure enums are cloned unchanged.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US housing inputs require the US schema.")
    person = frame.table("person")
    spm_unit = frame.table("spm_unit")
    _required_columns(
        person,
        ("person_spm_unit_id", "person_id"),
        "person",
    )
    _required_columns(
        spm_unit,
        (
            "spm_unit_id",
            "receives_housing_assistance",
            "takes_up_housing_assistance_if_eligible",
        ),
        "spm_unit",
    )
    if not has_support_role_metadata(person, entity="person"):
        raise ValueError("US housing PUF imputation requires person support metadata.")
    if not has_support_role_metadata(spm_unit, entity="spm_unit"):
        raise ValueError(
            "US housing PUF imputation requires SPM-unit support metadata."
        )
    predictors = _person_puf_predictors(frame)
    receipt_values, receipt_missing, receipt_invalid = _strict_boolean_signal(
        spm_unit["receives_housing_assistance"]
    )
    take_up_values, take_up_missing, take_up_invalid = _strict_boolean_signal(
        spm_unit["takes_up_housing_assistance_if_eligible"]
    )
    if receipt_missing or receipt_invalid or take_up_missing or take_up_invalid:
        raise ValueError(
            "US housing PUF imputation requires complete boolean receipt/take-up "
            "anchors; "
            f"receipt_missing={receipt_missing}, receipt_invalid={receipt_invalid}, "
            f"take_up_missing={take_up_missing}, take_up_invalid={take_up_invalid}."
        )
    mismatches = int(np.count_nonzero(receipt_values != take_up_values))
    if mismatches:
        raise ValueError(
            "US housing PUF imputation requires take-up to equal measured receipt "
            f"before fitting; mismatches={mismatches}."
        )
    receipt_by_unit = pd.Series(
        receipt_values,
        index=spm_unit["spm_unit_id"].to_numpy(),
    )
    person_target = person["person_spm_unit_id"].map(receipt_by_unit)
    if person_target.isna().any():
        bad = person.loc[person_target.isna(), "person_spm_unit_id"].head().tolist()
        raise ValueError(
            f"US housing assistance target does not cover person SPM unit id(s) {bad}."
        )

    role = support_role_series(person, entity="person")
    asec_mask = role == BASE_ASEC_SUPPORT_CHANNEL
    puf_mask = role == PUF_TAX_DETAIL_SUPPORT_CHANNEL
    if not asec_mask.any() or not puf_mask.any():
        raise ValueError(
            "US housing PUF imputation requires nonempty ASEC and PUF-tax-detail "
            "support channels."
        )
    predictor_names = tuple(predictors.columns)
    training = predictors.loc[asec_mask].copy()
    training["receives_housing_assistance"] = person_target.loc[asec_mask].to_numpy(
        dtype=np.float64
    )
    weights = pd.Series(
        np.asarray(frame.resolve_weights("person").values, dtype=np.float64),
        index=person.index,
    ).loc[asec_mask]
    if not np.isfinite(weights.to_numpy()).all() or (weights < 0.0).any():
        raise ValueError(
            "US housing PUF QRF person weights must be finite/nonnegative."
        )
    if float(weights.sum()) <= 0.0:
        raise ValueError("US housing PUF QRF person weights sum to zero.")
    if len(training) > int(max_train_samples):
        selected = training.sample(
            n=int(max_train_samples), random_state=int(seed)
        ).index
        training = training.loc[selected]
        weights = weights.loc[selected]
    test = predictors.loc[puf_mask].copy()
    for column in (*predictor_names, "receives_housing_assistance"):
        training[column] = pd.to_numeric(training[column], errors="coerce")
        if not np.isfinite(training[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"US housing PUF QRF training column {column!r} is nonfinite."
            )
    for column in predictor_names:
        test[column] = pd.to_numeric(test[column], errors="coerce")
        if not np.isfinite(test[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(
                f"US housing PUF QRF prediction column {column!r} is nonfinite."
            )

    global QRF
    if QRF is None:
        from importlib import import_module

        QRF = import_module("microcosm.fit").QRF
    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        training,
        predictors=list(predictor_names),
        targets=["receives_housing_assistance"],
        weights=weights.to_numpy(dtype=np.float64),
    )
    predicted = pd.to_numeric(
        fitted.predict(test)["receives_housing_assistance"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(predicted).all():
        raise ValueError("US housing PUF QRF produced nonfinite predictions.")

    person_result = person_target.astype(bool).copy()
    person_result.loc[puf_mask] = predicted >= 0.5
    unit_values = (
        pd.DataFrame(
            {
                "spm_unit_id": person["person_spm_unit_id"].to_numpy(),
                "value": person_result.to_numpy(dtype=bool),
            }
        )
        .groupby("spm_unit_id", sort=False)["value"]
        .first()
    )
    aligned = unit_values.reindex(spm_unit["spm_unit_id"])
    if aligned.isna().any():
        raise ValueError("US housing PUF QRF output does not cover every SPM unit.")
    puf_units = support_role_series(spm_unit, entity="spm_unit").eq(
        PUF_TAX_DETAIL_SUPPORT_CHANNEL
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["spm_unit"].loc[puf_units, "receives_housing_assistance"] = aligned.to_numpy(
        dtype=bool
    )[puf_units.to_numpy()]
    tables["spm_unit"].loc[puf_units, "takes_up_housing_assistance_if_eligible"] = (
        aligned.to_numpy(dtype=bool)[puf_units.to_numpy()]
    )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def with_us_housing_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    acs_rent_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> Frame:
    """Materialize source-backed housing and tenure inputs on a US frame."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US housing inputs require the US schema.")
    del time_period  # The archived donor vintage is pinned in the stage spec.
    us_housing_inputs_stage_spec()
    if us_housing_inputs_signal_gate(frame).passed:
        return frame
    carried = derive_us_housing_inputs(frame)
    rent = impute_us_pre_subsidy_rent(
        carried,
        acs_rent_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    tables = {entity: carried.table(entity).copy() for entity in carried.entities}
    tables["person"]["pre_subsidy_rent"] = rent
    return Frame(
        tables,
        carried.schema,
        {entity: carried.weights_for(entity) for entity in carried.weighted_entities},
        carried.strata,
        mass_log=carried.mass_log,
        metadata=carried.metadata,
    )


def us_housing_inputs_summary(frame: Frame) -> dict[str, object]:
    """Return weighted signal and entity-coherence diagnostics."""

    person = frame.table("person")
    household = frame.table("household")
    spm_unit = frame.table("spm_unit")
    rent = pd.to_numeric(person["pre_subsidy_rent"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    person_weights = np.asarray(
        frame.resolve_weights("person").values, dtype=np.float64
    )
    total_person_weight = float(person_weights.sum())
    positive_rent = np.isfinite(rent) & (rent > 0.0)
    rent_share = (
        float(person_weights[positive_rent].sum()) / total_person_weight
        if total_person_weight > 0.0
        else 0.0
    )

    head = person["is_household_head"].fillna(False).astype(bool).to_numpy()
    tenure_by_household = pd.Series(
        household["tenure_type"].astype(str).to_numpy(),
        index=household["household_id"].to_numpy(),
    )
    person_tenure = person["person_household_id"].map(tenure_by_household).astype(str)
    nonrenter_positive = positive_rent & person_tenure.ne("RENTED").to_numpy()

    receives, receives_missing, receives_invalid = _strict_boolean_signal(
        spm_unit["receives_housing_assistance"]
    )
    takes_up, takes_up_missing, takes_up_invalid = _strict_boolean_signal(
        spm_unit["takes_up_housing_assistance_if_eligible"]
    )
    spm_weights = np.asarray(frame.resolve_weights("spm_unit").values, dtype=np.float64)
    total_spm_weight = float(spm_weights.sum())
    receives_share = (
        float(spm_weights[receives].sum()) / total_spm_weight
        if total_spm_weight > 0.0
        else 0.0
    )
    assistance_share_by_channel: dict[str, float] = {}
    assistance_positive_by_channel: dict[str, int] = {}
    has_spm_support_metadata = has_support_role_metadata(
        spm_unit,
        entity="spm_unit",
    )
    if has_spm_support_metadata:
        channels = support_role_series(spm_unit, entity="spm_unit").to_numpy()
        for channel_name in (
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ):
            channel_mask = channels == channel_name
            if not channel_mask.any():
                continue
            channel_weight = float(spm_weights[channel_mask].sum())
            assistance_share_by_channel[channel_name] = (
                float(spm_weights[channel_mask & receives].sum()) / channel_weight
                if channel_weight > 0.0
                else 0.0
            )
            assistance_positive_by_channel[channel_name] = int(
                np.count_nonzero(channel_mask & receives)
            )
    household_values = sorted(set(household["tenure_type"].dropna().astype(str)))
    spm_values = sorted(set(spm_unit["spm_unit_tenure_type"].dropna().astype(str)))
    return {
        "pre_subsidy_rent_share": rent_share,
        "pre_subsidy_rent_share_band": list(_RENT_SHARE_BAND),
        "pre_subsidy_rent_total": float(np.nansum(rent * person_weights)),
        "pre_subsidy_rent_unweighted_total": float(np.nansum(rent)),
        "housing_assistance_share": receives_share,
        "housing_assistance_share_band": list(_HOUSING_ASSISTANCE_SHARE_BAND),
        "housing_assistance_receipt_missing_count": receives_missing,
        "housing_assistance_receipt_invalid_count": receives_invalid,
        "housing_assistance_take_up_missing_count": takes_up_missing,
        "housing_assistance_take_up_invalid_count": takes_up_invalid,
        "housing_assistance_take_up_source_mismatch_count": int(
            np.count_nonzero(takes_up != receives)
        ),
        "housing_assistance_share_by_support_channel": assistance_share_by_channel,
        "housing_assistance_positive_by_support_channel": (
            assistance_positive_by_channel
        ),
        "has_spm_support_channel_metadata": has_spm_support_metadata,
        "nonfinite_rent": int(np.count_nonzero(~np.isfinite(rent))),
        "negative_rent": int(np.count_nonzero(rent < 0.0)),
        "positive_rent_nonhead": int(np.count_nonzero(positive_rent & ~head)),
        "positive_rent_nonrenter": int(np.count_nonzero(nonrenter_positive)),
        "household_tenure_values": household_values,
        "spm_tenure_values": spm_values,
        "unknown_household_tenure_values": sorted(
            set(household_values) - _HOUSEHOLD_TENURE_VALUES
        ),
        "unknown_spm_tenure_values": sorted(set(spm_values) - _SPM_TENURE_VALUES),
    }


def us_housing_inputs_signal_gate(frame: Frame) -> GateResult:
    """Require plausible non-default signal for all five restored inputs."""

    missing: list[str] = []
    for entity, columns in (
        ("person", US_HOUSING_PERSON_OUTPUT_COLUMNS),
        ("spm_unit", US_HOUSING_SPM_UNIT_OUTPUT_COLUMNS),
        ("household", US_HOUSING_HOUSEHOLD_OUTPUT_COLUMNS),
    ):
        table = frame.table(entity)
        missing.extend(
            f"{entity}.{column}" for column in columns if column not in table.columns
        )
    if missing:
        return GateResult(
            name="housing_inputs_signal",
            passed=False,
            failures=(f"columns missing: {missing}.",),
            details={"missing": missing},
        )
    person = frame.table("person")
    required_person = ("is_household_head", "person_household_id")
    missing_support = [column for column in required_person if column not in person]
    if missing_support:
        return GateResult(
            name="housing_inputs_signal",
            passed=False,
            failures=(f"person support columns missing: {missing_support}.",),
            details={"missing": missing_support},
        )

    summary = us_housing_inputs_summary(frame)
    failures: list[str] = []
    for key, label in (
        (
            "housing_assistance_receipt_missing_count",
            "receives_housing_assistance missing values",
        ),
        (
            "housing_assistance_receipt_invalid_count",
            "receives_housing_assistance invalid non-boolean values",
        ),
        (
            "housing_assistance_take_up_missing_count",
            "takes_up_housing_assistance_if_eligible missing values",
        ),
        (
            "housing_assistance_take_up_invalid_count",
            "takes_up_housing_assistance_if_eligible invalid non-boolean values",
        ),
    ):
        count = int(summary[key])
        if count:
            failures.append(f"{label}: {count}.")
    take_up_mismatches = int(
        summary["housing_assistance_take_up_source_mismatch_count"]
    )
    if take_up_mismatches:
        failures.append(
            "takes_up_housing_assistance_if_eligible differs from measured "
            f"receives_housing_assistance on {take_up_mismatches} SPM unit(s)."
        )
    for key, label in (
        ("nonfinite_rent", "nonfinite pre_subsidy_rent"),
        ("negative_rent", "negative pre_subsidy_rent"),
        ("positive_rent_nonhead", "positive rent on non-head people"),
    ):
        count = int(summary[key])
        if count:
            failures.append(f"{label}: {count}.")
    for share_key, band_key, label in (
        (
            "pre_subsidy_rent_share",
            "pre_subsidy_rent_share_band",
            "positive pre-subsidy-rent share",
        ),
        (
            "housing_assistance_share",
            "housing_assistance_share_band",
            "housing-assistance receipt share",
        ),
    ):
        share = float(summary[share_key])
        low, high = summary[band_key]
        if not (low <= share <= high):
            failures.append(
                f"{label} {share:.3f} outside plausibility band [{low}, {high}]."
            )
    channel_shares = summary["housing_assistance_share_by_support_channel"]
    if summary["has_spm_support_channel_metadata"]:
        if PUF_TAX_DETAIL_SUPPORT_CHANNEL not in channel_shares:
            failures.append(
                "housing assistance support metadata has no PUF-tax-detail channel."
            )
        else:
            puf_share = float(channel_shares[PUF_TAX_DETAIL_SUPPORT_CHANNEL])
            low, high = _HOUSING_ASSISTANCE_SHARE_BAND
            if not (low <= puf_share <= high):
                failures.append(
                    "PUF-tax-detail housing-assistance receipt share "
                    f"{puf_share:.3f} outside plausibility band [{low}, {high}]."
                )
    household_values = summary["household_tenure_values"]
    spm_values = summary["spm_tenure_values"]
    if set(household_values) != _EXPECTED_HOUSEHOLD_TENURE_VALUES:
        failures.append(
            "household tenure_type does not carry the three locked ASEC "
            f"categories: {household_values}."
        )
    if set(spm_values) != _EXPECTED_SPM_TENURE_VALUES:
        failures.append(
            "spm_unit_tenure_type does not carry the three locked ASEC "
            f"categories: {spm_values}."
        )
    if summary["unknown_household_tenure_values"]:
        failures.append(
            "unknown household tenure value(s): "
            f"{summary['unknown_household_tenure_values']}."
        )
    if summary["unknown_spm_tenure_values"]:
        failures.append(
            f"unknown SPM tenure value(s): {summary['unknown_spm_tenure_values']}."
        )
    return GateResult(
        name="housing_inputs_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
