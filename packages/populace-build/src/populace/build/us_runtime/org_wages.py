"""CPS ORG labor-market inputs and the FLSA overtime-premium proxy.

This stage ports the final retired eCPS implementation at
``9a823603e6b5fb916d65ec45d74c9c7eb0043db1``.  It builds a person-level
training donor from the twelve official 2024 CPS basic-month outgoing-
rotation-group files, fits the weighted Populace QRF for hourly wage and
hourly-pay status, assigns union coverage from the published 2024 BLS state
rates, carries the ASEC occupation fields, and derives the intentionally
misspelled PolicyEngine input ``fsla_overtime_premium``.

The donor build, QRF, union assignment, and occupation carries remain the
retired port, with one recipient-side concept alignment: the QRF income
feature is the full-year-equivalent of actual annual income (income x
52/weeks, untouched when weeks are zero or missing), because the donor's
income is the annualized reference week — without the alignment, part-year
workers matched low-wage full-year donors and the imputed sub-minimum wage
tail was severe (populace#529).  The premium derivation deliberately deviates from the
retired reference-week-only construction: it adds the usual-weekly-hours
signal (ASEC HRSWK) as the persistent-overtime leg, so regular overtime
workers whose March reference week happened to sit at or below the threshold
are no longer dropped, and a single hot reference week no longer overrides a
measured above-threshold usual schedule.  On ASEC-channel rows HRSWK shares
the retrospective year with employment income and weeks worked (the reference
week does not); on PUF-support-channel rows both hours signals are donor-ASEC
schedule proxies attached to PUF income — the support design's property,
inherited equally by the retired reference-week leg.  Measured on the
certified Build N default, the reference-week-only gate carried 19.9M
weighted persons (19.0M tax units) against Treasury's published ">29 million"
TY2025 No-Tax-on-Overtime claimant floor; the two-signal estimator carries
the measured regular-overtime population the snapshot lottery missed.  Full
adjudication receipts: populace#451 item 4 (overtime-incidence lane).

The upstream implementation cached the transformed monthly files as
``census_cps_org_2024_wages.csv.gz`` but did not publish that cache.  Populace
pins the canonical *uncompressed CSV content* of the generated 119,237-row
donor.  A cached file is accepted only when that digest matches; rebuilding
from a silently reissued Census file therefore fails rather than changing a
release.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import urllib.request
import zipfile
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "BLS_STATE_UNION_REPRESENTATION_RATE_2024",
    "FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES",
    "FLSA_OVERTIME_OCCUPATION_CODES",
    "ORG_2024_DONOR_CONTENT_SHA256",
    "ORG_2024_DONOR_FILENAME",
    "ORG_PREDICTORS",
    "US_ORG_WAGES_NONCONSTANT_PERSON_COLUMNS",
    "US_ORG_WAGES_OUTPUT_COLUMNS",
    "US_ORG_WAGES_REQUIRED_SOURCE_COLUMNS",
    "US_ORG_WAGES_STAGE_NAME",
    "derive_flsa_overtime_premium",
    "derive_us_org_occupation_inputs",
    "fetch_org_2024_donor",
    "impute_us_org_wages",
    "load_org_2024_donor",
    "us_org_wages_signal_gate",
    "us_org_wages_stage_spec",
    "us_org_wages_summary",
    "with_us_org_wages_inputs",
]

US_ORG_WAGES_STAGE_NAME = "org_wages"
ORG_2024_DONOR_FILENAME = "census_cps_org_2024_wages.csv.gz"
ORG_2024_DONOR_CONTENT_SHA256 = (
    "d74600236cfdd34033d487cd9d82f6eb00b1858ba28de17f4f985f2aec516f86"
)
ORG_2024_DONOR_COMPRESSED_SHA256 = (
    "66fa5b6aa4087413b691038767b51f603281ff55411b58259922f78e67460372"
)
ORG_YEAR = 2024
ORG_MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)

ORG_PREDICTORS: tuple[str, ...] = (
    "employment_income",
    "weekly_hours_worked",
    "age",
    "is_female",
    "is_hispanic",
    "race_wbho",
    "state_fips",
)
ORG_QRF_TARGETS: tuple[str, ...] = ("hourly_wage", "is_paid_hourly")
_DONOR_WEIGHT_COLUMN = "sample_weight"
_DEFAULT_N_ESTIMATORS = 100

US_ORG_WAGES_OUTPUT_COLUMNS: tuple[str, ...] = (
    "cps_race",
    "is_hispanic",
    "detailed_occupation_recode",
    "has_never_worked",
    "is_military",
    "is_computer_scientist",
    "is_executive_administrative_professional",
    "is_farmer_fisher",
    "hourly_wage",
    "is_paid_hourly",
    "is_union_member_or_covered",
    "fsla_overtime_premium",
)
US_ORG_WAGES_NONCONSTANT_PERSON_COLUMNS = US_ORG_WAGES_OUTPUT_COLUMNS
US_ORG_WAGES_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_household_id",
    "age",
    "is_female",
    "PRDTRACE",
    "PRDTHSP",
    "POCCU2",
    "employment_income_before_lsr",
    "weekly_hours_worked_before_lsr",
    "hours_worked_last_week",
    "weeks_worked",
)

FLSA_OVERTIME_OCCUPATION_CODES: dict[str, int] = {
    "has_never_worked": 53,
    "is_military": 52,
    "is_computer_scientist": 8,
    "is_farmer_fisher": 41,
}
FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES = frozenset(
    (
        1,
        2,
        3,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        19,
        25,
        26,
        27,
        28,
        29,
        34,
        36,
        38,
        39,
        40,
        42,
        50,
    )
)

_CPS_BASIC_MONTHLY_ORG_COLUMNS = (
    "HRMIS",
    "gestfips",
    "prtage",
    "pesex",
    "ptdtrace",
    "pehspnon",
    "pworwgt",
    "pternwa",
    "pternhly",
    "peernhry",
    "pehruslt",
    "prerelg",
    "pemlr",
    "peio1cow",
)
_CPS_BASIC_MONTHLY_ORG_FWF_COLUMNS = (
    "HRMIS",
    "gestfips",
    "prtage",
    "pesex",
    "ptdtrace",
    "pehspnon",
    "pemlr",
    "pehruslt",
    "peio1cow",
    "prerelg",
    "peernhry",
    "pternhly",
    "pternwa",
    "pworwgt",
)
_CPS_BASIC_MONTHLY_ORG_COLSPECS = (
    (62, 64),
    (92, 94),
    (121, 123),
    (128, 130),
    (138, 140),
    (156, 158),
    (179, 181),
    (223, 226),
    (430, 433),
    (497, 499),
    (505, 507),
    (519, 523),
    (526, 534),
    (602, 612),
)

BLS_UNION_REPRESENTATION_RATE_2024 = np.float32(0.111)
_BLS_SEX_RATES = {False: np.float32(0.113), True: np.float32(0.108)}
_BLS_AGE_RATES = {
    (16, 24): np.float32(0.054),
    (25, 34): np.float32(0.099),
    (35, 44): np.float32(0.122),
    (45, 54): np.float32(0.138),
    (55, 64): np.float32(0.130),
    (65, 150): np.float32(0.102),
}
_BLS_RACE_RATES = {
    1: np.float32(0.108),
    2: np.float32(0.132),
    3: np.float32(0.097),
    4: BLS_UNION_REPRESENTATION_RATE_2024,
}
_BLS_FULL_TIME_RATE = np.float32(0.120)
_BLS_PART_TIME_RATE = np.float32(0.066)
BLS_STATE_UNION_REPRESENTATION_RATE_2024: dict[int, np.float32] = {
    1: np.float32(0.078),
    2: np.float32(0.195),
    4: np.float32(0.045),
    5: np.float32(0.044),
    6: np.float32(0.163),
    8: np.float32(0.080),
    9: np.float32(0.178),
    10: np.float32(0.089),
    11: np.float32(0.117),
    12: np.float32(0.063),
    13: np.float32(0.044),
    15: np.float32(0.275),
    16: np.float32(0.059),
    17: np.float32(0.142),
    18: np.float32(0.104),
    19: np.float32(0.083),
    20: np.float32(0.080),
    21: np.float32(0.112),
    22: np.float32(0.050),
    23: np.float32(0.153),
    24: np.float32(0.134),
    25: np.float32(0.156),
    26: np.float32(0.147),
    27: np.float32(0.148),
    28: np.float32(0.079),
    29: np.float32(0.093),
    30: np.float32(0.131),
    31: np.float32(0.081),
    32: np.float32(0.134),
    33: np.float32(0.106),
    34: np.float32(0.174),
    35: np.float32(0.088),
    36: np.float32(0.219),
    37: np.float32(0.031),
    38: np.float32(0.063),
    39: np.float32(0.133),
    40: np.float32(0.062),
    41: np.float32(0.175),
    42: np.float32(0.124),
    44: np.float32(0.153),
    45: np.float32(0.041),
    46: np.float32(0.037),
    47: np.float32(0.056),
    48: np.float32(0.054),
    49: np.float32(0.078),
    50: np.float32(0.158),
    51: np.float32(0.057),
    53: np.float32(0.183),
    54: np.float32(0.100),
    55: np.float32(0.069),
    56: np.float32(0.067),
}

_SHARE_BANDS: dict[str, tuple[float, float]] = {
    "cps_race": (0.95, 1.0),
    "is_hispanic": (0.05, 0.30),
    "detailed_occupation_recode": (0.65, 0.95),
    "has_never_worked": (0.15, 0.35),
    "is_military": (0.0005, 0.01),
    "is_computer_scientist": (0.005, 0.05),
    "is_executive_administrative_professional": (0.20, 0.40),
    "is_farmer_fisher": (0.001, 0.015),
    "hourly_wage": (0.40, 0.70),
    "is_paid_hourly": (0.15, 0.40),
    "is_union_member_or_covered": (0.02, 0.12),
    "fsla_overtime_premium": (0.01, 0.15),
}


def us_org_wages_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged ``org_wages`` declaration."""

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map().get(US_ORG_WAGES_STAGE_NAME)
    if spec is None:
        raise ValueError("US source manifest declares no 'org_wages' stage.")
    missing = sorted(set(US_ORG_WAGES_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"'org_wages' manifest stage does not declare output(s) {missing}."
        )
    return spec


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_donor_bytes(path: str | Path) -> bytes:
    source = Path(path)
    if source.suffix == ".gz":
        return gzip.decompress(source.read_bytes())
    return source.read_bytes()


def _month_url(month: str, suffix: str) -> str:
    return (
        "https://www2.census.gov/programs-surveys/cps/datasets/"
        f"{ORG_YEAR}/basic/{month}24pub.{suffix}"
    )


def _normalized_month(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).lower(): column for column in frame.columns}
    missing = [
        column
        for column in _CPS_BASIC_MONTHLY_ORG_COLUMNS
        if column.lower() not in lookup
    ]
    if missing:
        raise ValueError(f"CPS basic ORG month missing column(s): {missing}.")
    selected = frame[
        [lookup[column.lower()] for column in _CPS_BASIC_MONTHLY_ORG_COLUMNS]
    ].copy()
    selected.columns = _CPS_BASIC_MONTHLY_ORG_COLUMNS
    return selected.apply(pd.to_numeric, errors="coerce")


def _load_month_from_network(month: str) -> pd.DataFrame:
    """Load one official month, preferring CSV and falling back to its ZIP."""

    csv_error: Exception | None = None
    try:
        with urllib.request.urlopen(  # noqa: S310
            _month_url(month, "csv"), timeout=60
        ) as response:
            return _normalized_month(pd.read_csv(io.BytesIO(response.read())))
    except Exception as error:  # pragma: no cover - live source fallback
        csv_error = error
    try:
        with urllib.request.urlopen(  # noqa: S310
            _month_url(month, "zip"), timeout=60
        ) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next(
                name for name in archive.namelist() if name.lower().endswith(".dat")
            )
            with archive.open(member) as stream:
                frame = pd.read_fwf(
                    stream,
                    colspecs=_CPS_BASIC_MONTHLY_ORG_COLSPECS,
                    names=_CPS_BASIC_MONTHLY_ORG_FWF_COLUMNS,
                    header=None,
                )
        return _normalized_month(frame)
    except Exception as zip_error:  # pragma: no cover - live source failure
        raise ValueError(
            f"Could not load CPS ORG {month} {ORG_YEAR} from CSV or ZIP: "
            f"csv={csv_error!r}; zip={zip_error!r}."
        ) from zip_error


def _derive_wbho(cps_race: np.ndarray, is_hispanic: np.ndarray) -> np.ndarray:
    hispanic = np.asarray(is_hispanic, dtype=bool)
    race = np.asarray(cps_race)
    return np.select(
        [hispanic, (race == 1) & ~hispanic, (race == 2) & ~hispanic],
        [3, 1, 2],
        default=4,
    ).astype(np.float32)


def _transform_month(raw: pd.DataFrame) -> pd.DataFrame:
    org = raw.loc[
        raw["HRMIS"].isin([4, 8])
        & (raw["pworwgt"] > 0)
        & (raw["prerelg"] == 1)
        & raw["pemlr"].isin([1, 2])
        & raw["peio1cow"].isin([1, 2, 3, 4, 5])
        & raw["peernhry"].isin([1, 2])
        & (raw["gestfips"] > 0)
        & (raw["prtage"] >= 16)
        & (raw["pehruslt"] > 0)
        & (raw["pternwa"] > 0)
    ].copy()
    weekly = org["pternwa"].to_numpy(dtype=np.float32) / np.float32(100.0)
    hours = org["pehruslt"].to_numpy(dtype=np.float32)
    direct_hourly = org["pternhly"].to_numpy(dtype=np.float32) / np.float32(100.0)
    hispanic = (org["pehspnon"].to_numpy() == 1).astype(np.float32)
    result = pd.DataFrame(
        {
            "employment_income": weekly * np.float32(52.0),
            "weekly_hours_worked": hours,
            "age": org["prtage"].to_numpy(dtype=np.float32),
            "is_female": (org["pesex"].to_numpy() == 2).astype(np.float32),
            "is_hispanic": hispanic,
            "race_wbho": _derive_wbho(org["ptdtrace"].to_numpy(), hispanic),
            "state_fips": org["gestfips"].to_numpy(dtype=np.float32),
            "hourly_wage": np.where(
                (org["peernhry"].to_numpy() == 1) & (direct_hourly > 0),
                direct_hourly,
                weekly / hours,
            ),
            "is_paid_hourly": (org["peernhry"].to_numpy() == 1).astype(np.float32),
            _DONOR_WEIGHT_COLUMN: org["pworwgt"].to_numpy(dtype=np.float32),
        }
    )
    return result.loc[
        (result["employment_income"] > 0)
        & (result["weekly_hours_worked"] > 0)
        & (result["hourly_wage"] > 0)
    ].reset_index(drop=True)


def fetch_org_2024_donor(
    cache_dir: str | Path | None = None,
    *,
    expected_content_sha256: str | None = ORG_2024_DONOR_CONTENT_SHA256,
) -> Path:
    """Build/cache the exact twelve-month ORG donor and verify its content."""

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache" / "populace" / "org"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / ORG_2024_DONOR_FILENAME
    if target.exists():
        try:
            content = _canonical_donor_bytes(target)
        except OSError:
            content = b""
        if content and (
            expected_content_sha256 is None
            or _sha256(content) == expected_content_sha256
        ):
            return target

    donor = pd.concat(
        [_transform_month(_load_month_from_network(month)) for month in ORG_MONTHS],
        ignore_index=True,
    )
    content = donor.to_csv(index=False).encode("utf-8")
    digest = _sha256(content)
    if expected_content_sha256 is not None and digest != expected_content_sha256:
        raise ValueError(
            "CPS ORG donor failed canonical sha-256 verification after rebuilding: "
            f"expected {expected_content_sha256}, got {digest}."
        )
    target.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
    return target


def load_org_2024_donor(
    path: str | Path,
    *,
    expected_content_sha256: str | None = None,
) -> pd.DataFrame:
    """Load a transformed ORG donor, optionally verifying canonical bytes."""

    content = _canonical_donor_bytes(path)
    if expected_content_sha256 is not None:
        digest = _sha256(content)
        if digest != expected_content_sha256:
            raise ValueError(
                "CPS ORG donor failed canonical sha-256 verification: "
                f"expected {expected_content_sha256}, got {digest}."
            )
    donor = pd.read_csv(io.BytesIO(content))
    required = [*ORG_PREDICTORS, *ORG_QRF_TARGETS, _DONOR_WEIGHT_COLUMN]
    missing = [column for column in required if column not in donor]
    if missing:
        raise ValueError(f"CPS ORG donor missing column(s): {missing}.")
    donor = donor.loc[:, required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(donor.to_numpy(dtype=np.float64)).all(axis=1)
    valid = (
        finite
        & (donor[_DONOR_WEIGHT_COLUMN].to_numpy() > 0)
        & (donor["employment_income"].to_numpy() > 0)
        & (donor["weekly_hours_worked"].to_numpy() > 0)
        & (donor["hourly_wage"].to_numpy() > 0)
    )
    result = donor.loc[valid].reset_index(drop=True)
    if result.empty:
        raise ValueError("CPS ORG donor has no valid training rows.")
    return result


def derive_us_org_occupation_inputs(person: pd.DataFrame) -> pd.DataFrame:
    """Carry CPS race/ethnicity and derive the exact POCCU2 FLSA flags."""

    missing = [
        column for column in ("PRDTRACE", "PRDTHSP", "POCCU2") if column not in person
    ]
    if missing:
        raise ValueError(
            f"ORG occupation derivation needs person column(s): {missing}."
        )
    occupation = pd.to_numeric(person["POCCU2"], errors="coerce").fillna(0).astype(int)
    result = pd.DataFrame(index=person.index)
    result["cps_race"] = (
        pd.to_numeric(person["PRDTRACE"], errors="coerce").fillna(0).astype(np.int16)
    )
    result["is_hispanic"] = (
        pd.to_numeric(person["PRDTHSP"], errors="coerce").fillna(0).ne(0)
    )
    result["detailed_occupation_recode"] = occupation.astype(np.int16)
    for variable, code in FLSA_OVERTIME_OCCUPATION_CODES.items():
        result[variable] = occupation.eq(code)
    result["is_executive_administrative_professional"] = occupation.isin(
        FLSA_EXECUTIVE_ADMINISTRATIVE_PROFESSIONAL_OCCUPATION_CODES
    )
    return result


def _person_state_fips(frame: Frame) -> np.ndarray:
    person = frame.table("person")
    if "state_fips" in person:
        return pd.to_numeric(person["state_fips"], errors="coerce").fillna(0).to_numpy()
    household = frame.table("household")
    if "state_fips" not in household:
        raise ValueError("ORG imputation needs household.state_fips.")
    mapping = pd.Series(
        household["state_fips"].to_numpy(),
        index=household["household_id"].to_numpy(),
    )
    state = person["person_household_id"].map(mapping)
    if state.isna().any():
        raise ValueError(
            "ORG imputation could not map every person to household state_fips."
        )
    return pd.to_numeric(state, errors="coerce").fillna(0).to_numpy()


def _recipient_features(frame: Frame, carried: pd.DataFrame) -> pd.DataFrame:
    person = frame.table("person")
    missing = [
        column
        for column in US_ORG_WAGES_REQUIRED_SOURCE_COLUMNS
        if column not in person
    ]
    if missing:
        raise ValueError(f"ORG imputation needs person column(s): {missing}.")
    features = pd.DataFrame(index=person.index)
    annual_income = (
        pd.to_numeric(person["employment_income_before_lsr"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=np.float64)
    )
    weeks = (
        pd.to_numeric(person["weeks_worked"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=52)
        .to_numpy(dtype=np.float64)
    )
    # The donor's employment_income is the annualized reference week
    # (pternwa x 52), while recipients carry actual annual income. Feeding
    # the full-year-equivalent aligns the concepts so part-year workers
    # match same-wage donors instead of low-wage full-year ones; zero or
    # missing weeks pass income through untouched (populace#529). The
    # premium derivation keeps actual annual income.
    factor = np.divide(52.0, weeks, out=np.ones_like(weeks), where=weeks > 0)
    features["employment_income"] = annual_income * factor
    features["weekly_hours_worked"] = (
        pd.to_numeric(person["weekly_hours_worked_before_lsr"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    features["age"] = pd.to_numeric(person["age"], errors="coerce").fillna(0)
    features["is_female"] = pd.to_numeric(person["is_female"], errors="coerce").fillna(
        0
    )
    features["is_hispanic"] = carried["is_hispanic"].astype(np.float64)
    features["race_wbho"] = _derive_wbho(
        carried["cps_race"].to_numpy(), carried["is_hispanic"].to_numpy()
    )
    features["state_fips"] = _person_state_fips(frame)
    return features.loc[:, list(ORG_PREDICTORS)]


def _union_priority(features: pd.DataFrame) -> np.ndarray:
    base = float(BLS_UNION_REPRESENTATION_RATE_2024)
    age = features["age"].to_numpy(dtype=np.float64)
    age_rate = np.full(len(features), base)
    for (low, high), rate in _BLS_AGE_RATES.items():
        age_rate[(age >= low) & (age <= high)] = rate
    sex_rate = np.where(
        features["is_female"].to_numpy() >= 0.5,
        _BLS_SEX_RATES[True],
        _BLS_SEX_RATES[False],
    )
    race = features["race_wbho"].to_numpy(dtype=int)
    race_rate = np.full(len(features), base)
    for code, rate in _BLS_RACE_RATES.items():
        race_rate[race == code] = rate
    hours_rate = np.where(
        features["weekly_hours_worked"].to_numpy() >= 35,
        _BLS_FULL_TIME_RATE,
        _BLS_PART_TIME_RATE,
    )
    return np.clip(
        (age_rate / base)
        * (sex_rate / base)
        * (race_rate / base)
        * (hours_rate / base),
        1e-6,
        None,
    )


def _assign_union(features: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(features), dtype=bool)
    eligible = (
        (features["employment_income"].to_numpy() > 0)
        & (features["weekly_hours_worked"].to_numpy() > 0)
        & (features["age"].to_numpy() >= 16)
    )
    if not eligible.any():
        return result
    hashes = pd.util.hash_pandas_object(
        pd.DataFrame(
            {
                column: np.round(features[column].to_numpy(dtype=np.float64), 4)
                for column in ORG_PREDICTORS
            }
        ),
        index=False,
    ).to_numpy(dtype=np.uint64)
    uniform = (hashes.astype(np.float64) + 0.5) / (np.iinfo(np.uint64).max + 1.0)
    key = -np.log(np.clip(uniform, 1e-12, 1 - 1e-12)) / _union_priority(features)
    states = np.nan_to_num(features["state_fips"].to_numpy(), nan=-1).astype(int)
    for state in np.unique(states[eligible]):
        positions = np.flatnonzero(eligible & (states == state))
        rate = float(
            BLS_STATE_UNION_REPRESENTATION_RATE_2024.get(
                state, BLS_UNION_REPRESENTATION_RATE_2024
            )
        )
        count = min(len(positions), max(0, int(np.rint(rate * len(positions)))))
        if count:
            selected = np.argpartition(key[positions], count - 1)[:count]
            result[positions[selected]] = True
    return result


def impute_us_org_wages(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QRF-impute hourly inputs and return them with direct ASEC carries."""

    from populace.fit import QRF

    carried = derive_us_org_occupation_inputs(frame.table("person"))
    features = _recipient_features(frame, carried)
    required = [*ORG_PREDICTORS, *ORG_QRF_TARGETS, _DONOR_WEIGHT_COLUMN]
    missing = [column for column in required if column not in donor]
    if missing:
        raise ValueError(f"CPS ORG donor missing column(s): {missing}.")
    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        donor.loc[:, required],
        predictors=list(ORG_PREDICTORS),
        targets=list(ORG_QRF_TARGETS),
        weights=_DONOR_WEIGHT_COLUMN,
    )
    predicted = fitted.predict(features)
    inactive = (features["employment_income"].to_numpy() <= 0) | (
        features["weekly_hours_worked"].to_numpy() <= 0
    )
    wages = pd.DataFrame(index=features.index)
    wages["hourly_wage"] = np.maximum(
        pd.to_numeric(predicted["hourly_wage"], errors="coerce").fillna(0).to_numpy(), 0
    )
    wages["is_paid_hourly"] = (
        pd.to_numeric(predicted["is_paid_hourly"], errors="coerce").fillna(0).to_numpy()
        >= 0.5
    )
    wages["is_union_member_or_covered"] = _assign_union(features)
    wages.loc[inactive, "hourly_wage"] = 0.0
    wages.loc[inactive, ["is_paid_hourly", "is_union_member_or_covered"]] = False
    return carried, wages


def _finite_nonnegative(values: np.ndarray | pd.Series) -> np.ndarray:
    """Read a numeric signal with NaN and infinities treated as absent."""

    return np.maximum(
        np.nan_to_num(
            np.asarray(
                pd.to_numeric(pd.Series(values), errors="coerce"),
                dtype=np.float64,
            ),
            nan=0,
            posinf=0,
            neginf=0,
        ),
        0,
    )


def _flsa_policy(year: int) -> tuple[float, float, float, float, float]:
    from policyengine_us import CountryTaxBenefitSystem

    overtime = (
        CountryTaxBenefitSystem()
        .parameters(f"{int(year)}-01-01")
        .gov.irs.income.exemption.overtime
    )
    hours = float(overtime.hours_threshold)
    return (
        float(overtime.hce_salary_threshold),
        float(overtime.salary_basis_threshold) * 52.0,
        float(overtime.computer_salary_threshold) * hours * 52.0,
        hours,
        float(overtime.rate_multiplier),
    )


def derive_flsa_overtime_premium(
    *,
    time_period: int,
    employment_income: np.ndarray | pd.Series,
    hours_worked_last_week: np.ndarray | pd.Series,
    usual_weekly_hours: np.ndarray | pd.Series,
    weeks_worked: np.ndarray | pd.Series,
    is_paid_hourly: np.ndarray | pd.Series,
    has_never_worked: np.ndarray | pd.Series,
    is_military: np.ndarray | pd.Series,
    is_executive_administrative_professional: np.ndarray | pd.Series,
    is_farmer_fisher: np.ndarray | pd.Series,
    is_computer_scientist: np.ndarray | pd.Series,
    policy: tuple[float, float, float, float, float] | None = None,
) -> np.ndarray:
    """Two-signal annual-wage-share FLSA overtime proxy.

    The retired method annualized the ASEC reference week alone: a worker was
    a carrier only when the March survey week exceeded the hours threshold,
    and that single week's premium share was applied to the whole retrospective
    income year. That construction is a mass proxy — exact for steady
    schedules, biased either way for variable ones — that collapses incidence
    to one week's draw, and the reference week is not even inside the income
    year the premium is attributed to.

    This estimator adds the persistent signal: when usual weekly hours worked
    last year (ASEC HRSWK) exceed the threshold, the worker is a carrier and
    the usual-hours share is the annualizer — a single hot or absent reference
    week neither overrides nor erases a regular overtime schedule. On
    ASEC-channel rows HRSWK shares the retrospective year with
    ``employment_income`` and ``weeks_worked``; on PUF-support-channel rows
    both hours signals are donor-ASEC schedule proxies attached to PUF income
    (the support design's property, shared by the retired reference-week leg).
    Workers whose usual week stays at or below the threshold retain the
    retired reference-week leg unchanged: it is representative for steady
    schedules and can under- or over-recover an occasional overtimer's annual
    premium depending on the sampled week — not an unbiased mass estimator —
    but it is the only measured signal for that population, and replacing it
    would require modeled participation rather than measured hours. Both
    hours measures count all jobs while FLSA §7 applies per employer; that
    shared upward concept edge is inherited from the retired method.
    Exemption screens are unchanged. Malformed non-finite inputs (NaN or
    infinities) are treated as absent rather than annualized. Adjudication
    receipts: populace#451 item 4 (overtime-incidence lane).
    """

    income = _finite_nonnegative(employment_income)
    reference_week = _finite_nonnegative(hours_worked_last_week)
    usual = _finite_nonnegative(usual_weekly_hours)
    weeks = _finite_nonnegative(weeks_worked)
    paid_hourly = np.asarray(is_paid_hourly, dtype=bool)
    never = np.asarray(has_never_worked, dtype=bool)
    military = np.asarray(is_military, dtype=bool)
    eap = np.asarray(is_executive_administrative_professional, dtype=bool)
    farmer = np.asarray(is_farmer_fisher, dtype=bool)
    computer = np.asarray(is_computer_scientist, dtype=bool)
    hce, salary_basis, computer_salary, threshold_hours, multiplier = (
        _flsa_policy(time_period) if policy is None else policy
    )
    hours = np.where(usual > threshold_hours, usual, reference_week)
    overtime_hours = np.maximum(hours - threshold_hours, 0)
    straight_equivalent = (
        np.minimum(hours, threshold_hours) + overtime_hours * multiplier
    )
    premium_share = np.divide(
        (multiplier - 1) * overtime_hours,
        straight_equivalent,
        out=np.zeros_like(income),
        where=straight_equivalent > 0,
    )
    salary_threshold = np.full_like(income, hce)
    salary_threshold = np.where(computer, min(computer_salary, hce), salary_threshold)
    salary_threshold = np.where(eap | farmer, min(salary_basis, hce), salary_threshold)
    always_exempt = never | military
    exempt = always_exempt | ((income >= salary_threshold) & ~paid_hourly)
    premium = np.where(~exempt & (weeks > 0), income * premium_share, 0)
    return np.minimum(premium, income).astype(np.float32)


def _surface_has_signal(person: pd.DataFrame) -> bool:
    return all(
        column in person and person[column].dropna().nunique() > 1
        for column in US_ORG_WAGES_OUTPUT_COLUMNS
    )


def _outputs_match(person: pd.DataFrame, outputs: dict[str, np.ndarray]) -> bool:
    """True when every recomputed output equals the stored column exactly.

    Only boolean, integer, and float dtype families may match — anything
    else (object, string, temporal, categorical) always rebuilds, because a
    coercible representation ("1" strings, 0/1-ns datetimes) must not let a
    damaged dtype survive the fast path. Booleans compare through strict
    0/1 numerics so NaN and pd.NA rebuild rather than casting truthy;
    numerics compare as float32 (the shipped dtype) where a stored NaN never
    compares equal. A damaged column therefore always takes the recomputed
    clean values.
    """

    for column, values in outputs.items():
        if column not in person:
            return False
        stored_series = person[column]
        # Positive dtype-family allowlist: shipped surfaces carry only
        # boolean, integer, and float columns (verified against the cached
        # Build N/O stores). Rebuild on anything else.
        if not (
            pd.api.types.is_bool_dtype(stored_series)
            or pd.api.types.is_integer_dtype(stored_series)
            or pd.api.types.is_float_dtype(stored_series)
        ):
            return False
        if values.dtype == bool:
            # Compare through strict 0/1 numerics: NaN, pd.NA, or any
            # non-canonical value fails the membership test and takes the
            # rebuild path instead of casting to a truthy True.
            stored = pd.to_numeric(stored_series, errors="coerce").to_numpy(
                dtype=np.float64
            )
            if not np.isin(stored, (0.0, 1.0)).all():
                return False
            if not np.array_equal(stored.astype(bool), values):
                return False
        else:
            stored = pd.to_numeric(stored_series, errors="coerce").to_numpy(
                dtype=np.float32
            )
            if not np.array_equal(stored, np.asarray(values, dtype=np.float32)):
                return False
    return True


def with_us_org_wages_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    org_donor: pd.DataFrame,
) -> Frame:
    """Apply the complete ORG/occupation/FLSA family before calibration.

    The family is re-imputed on every entry. A frame whose ORG surface
    already matches the current estimator passes through unchanged — the
    seeded QRF, deterministic union lottery, and pure derives make same-code
    re-entry byte-stable — while any stale construction (wages, hourly
    status, union coverage, occupation carries, or the FLSA premium)
    converges instead of silently surviving.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US ORG inputs require the US schema.")
    us_org_wages_stage_spec()
    person = frame.table("person")
    carried, wages = impute_us_org_wages(frame, org_donor, seed=seed)
    premium = derive_flsa_overtime_premium(
        time_period=time_period,
        employment_income=person["employment_income_before_lsr"],
        hours_worked_last_week=person["hours_worked_last_week"],
        usual_weekly_hours=person["weekly_hours_worked_before_lsr"],
        weeks_worked=person["weeks_worked"],
        is_paid_hourly=wages["is_paid_hourly"],
        has_never_worked=carried["has_never_worked"],
        is_military=carried["is_military"],
        is_executive_administrative_professional=carried[
            "is_executive_administrative_professional"
        ],
        is_farmer_fisher=carried["is_farmer_fisher"],
        is_computer_scientist=carried["is_computer_scientist"],
    )
    outputs: dict[str, np.ndarray] = {}
    for column in carried:
        outputs[column] = carried[column].to_numpy()
    for column in wages:
        outputs[column] = wages[column].to_numpy()
    outputs["fsla_overtime_premium"] = premium
    if _surface_has_signal(person) and _outputs_match(person, outputs):
        return frame
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in outputs.items():
        tables["person"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_org_wages_summary(frame: Frame) -> dict[str, object]:
    """Weighted incidence plus structural-invariant diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())

    def share(column: str) -> float:
        values = pd.to_numeric(person[column], errors="coerce").fillna(0).to_numpy()
        return float(weights[values != 0].sum() / total_weight) if total_weight else 0.0

    premium = pd.to_numeric(person["fsla_overtime_premium"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    # Input signals share the estimator's non-finite-as-absent read, so the
    # structural checks below judge exactly what the derivation saw; the
    # premium itself stays raw so nonfinite outputs are counted, not masked.
    income = _finite_nonnegative(person["employment_income_before_lsr"])
    hours = _finite_nonnegative(person["hours_worked_last_week"])
    usual = _finite_nonnegative(person["weekly_hours_worked_before_lsr"])
    weeks = _finite_nonnegative(person["weeks_worked"])
    never = person["has_never_worked"].astype(bool).to_numpy()
    military = person["is_military"].astype(bool).to_numpy()
    violations = {
        "nonfinite": int((~np.isfinite(premium)).sum()),
        "negative": int((premium < 0).sum()),
        "exceeds_employment_income": int(
            (premium > np.maximum(income, 0) + 1e-4).sum()
        ),
        # A positive premium is structurally impossible only when neither
        # measured hours signal exceeds the threshold.
        "positive_without_overtime": int(
            ((premium > 0) & (hours <= 40) & (usual <= 40)).sum()
        ),
        "positive_without_weeks": int(((premium > 0) & (weeks <= 0)).sum()),
        "positive_always_exempt": int(((premium > 0) & (never | military)).sum()),
    }
    return {
        "nonzero_shares": {column: share(column) for column in _SHARE_BANDS},
        "share_bands": {column: list(band) for column, band in _SHARE_BANDS.items()},
        "unique_counts": {
            column: int(person[column].dropna().nunique())
            for column in US_ORG_WAGES_OUTPUT_COLUMNS
        },
        "fsla_overtime_premium_weighted_total": float(np.nansum(premium * weights)),
        "constraint_violations": violations,
    }


def us_org_wages_signal_gate(frame: Frame) -> GateResult:
    """Require the full ORG/FLSA family to carry plausible, coherent signal."""

    person = frame.table("person")
    missing = [column for column in US_ORG_WAGES_OUTPUT_COLUMNS if column not in person]
    if missing:
        return GateResult(
            name="org_wages_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )
    summary = us_org_wages_summary(frame)
    failures: list[str] = []
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(f"{column}: constant column carries no signal.")
    for column, band in _SHARE_BANDS.items():
        value = float(summary["nonzero_shares"][column])
        low, high = band
        if not (low <= value <= high):
            failures.append(
                f"{column}: weighted nonzero share {value:.4f} outside [{low}, {high}]."
            )
    for name, count in summary["constraint_violations"].items():
        if count:
            failures.append(f"fsla_overtime_premium: {count} {name} violation(s).")
    return GateResult(
        name="org_wages_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
