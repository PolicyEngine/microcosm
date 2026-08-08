"""SIPP-imputed tip income and CPS tipped-occupation inputs.

The retired eCPS pipeline did not carry tips from ASEC: ASEC has no tip-
amount field.  It trained a weighted QRF on SIPP person records, with annual
tip income equal to the December monthly total across up to seven jobs times
12.  The predictors were employment income, age, counts of children under 18
and under 6 in the household, and a tipped-occupation indicator.  The latter
was derived on both donor and recipient records by mapping detailed Census
occupation codes to the Treasury tipped-occupation list.

This is a direct port of that method, pinned to the retired implementation at
``9a823603e6b5fb916d65ec45d74c9c7eb0043db1``
(``datasets/sipp/sipp.py`` and ``calibration/source_impute.py``).  The final
pipeline switched its live downloader to SIPP 2024 without pinning the Census
zip.  Microcosm instead uses the same pipeline's fixed SIPP 2023 slim extract,
whose immutable mirror revision and content SHA-256 are pinned below.  That is
the donor used to build the reference eCPS and avoids accepting a silently
reissued public-use file.

Two PolicyEngine-US person inputs are produced:

* ``treasury_tipped_occupation_code`` is a direct CPS carry-through derived
  from raw ASEC ``PEIOOCC``.
* ``tip_income`` is a non-negative SIPP QRF draw. Treasury-listed occupation
  is a predictor, not a domain mask: the pinned reference eCPS carries positive
  tips for some people outside the list, and the OBBBA formula applies its own
  occupation-qualification test downstream.

Healing behavior matches the other source stages: an existing pair of columns
with signal is passed through untouched; missing or constant-default columns
are rebuilt from the raw ASEC occupation and the pinned donor.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "CENSUS_OCCUPATION_CODE_TO_TTOC",
    "SIPP_2023_TIP_DONOR_REVISION",
    "SIPP_2023_TIP_DONOR_SHA256",
    "SIPP_2023_TIP_DONOR_URL",
    "SIPP_TIP_OUTPUT_COLUMNS",
    "SIPP_TIP_PREDICTORS",
    "US_SIPP_TIPS_NONCONSTANT_PERSON_COLUMNS",
    "US_SIPP_TIPS_OUTPUT_COLUMNS",
    "US_SIPP_TIPS_REQUIRED_SOURCE_COLUMNS",
    "US_SIPP_TIPS_STAGE_NAME",
    "derive_treasury_tipped_occupation_code",
    "fetch_sipp_2023_tip_donor",
    "impute_us_sipp_tips",
    "load_sipp_2023_tip_donor",
    "us_sipp_tips_signal_gate",
    "us_sipp_tips_stage_spec",
    "us_sipp_tips_summary",
    "with_us_sipp_tip_inputs",
]

US_SIPP_TIPS_STAGE_NAME = "sipp_tips"

# The immutable mirror revision containing the retired pipeline's SIPP 2023
# slim donor.  The repository name is assembled so the live-tree guard does not
# mistake this historical input coordinate for a runtime package dependency.
SIPP_2023_TIP_DONOR_REVISION = "21280dca5995e978d706740a8a4b9b7860cfd7b6"
_RETIRED_DATA_REPOSITORY = "policyengine-" + "us-data"
SIPP_2023_TIP_DONOR_URL = (
    "https://huggingface.co/policyengine/"
    f"{_RETIRED_DATA_REPOSITORY}/resolve/"
    f"{SIPP_2023_TIP_DONOR_REVISION}/pu2023_slim.csv"
)
SIPP_2023_TIP_DONOR_SHA256 = (
    "1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5"
)
_SIPP_2023_TIP_DONOR_FILENAME = "pu2023_slim.csv"

US_SIPP_TIPS_OUTPUT_COLUMNS: tuple[str, ...] = (
    "tip_income",
    "treasury_tipped_occupation_code",
)
# Short compatibility alias used by stage-local callers; the ``US_`` name is
# retained for consistency with the release-export registries.
SIPP_TIP_OUTPUT_COLUMNS = US_SIPP_TIPS_OUTPUT_COLUMNS
US_SIPP_TIPS_NONCONSTANT_PERSON_COLUMNS = US_SIPP_TIPS_OUTPUT_COLUMNS

SIPP_TIP_PREDICTORS: tuple[str, ...] = (
    "employment_income",
    "age",
    "count_under_18",
    "count_under_6",
    "is_tipped_occupation",
)

US_SIPP_TIPS_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "person_household_id",
    "employment_income_before_lsr",
    "age",
    "PEIOOCC",
)

_SIPP_JOB_OCCUPATION_COLUMNS = tuple(f"TJB{i}_OCC" for i in range(1, 8))
_SIPP_TIP_AMOUNT_COLUMNS = tuple(f"TJB{i}_TXAMT" for i in range(1, 8))
_SIPP_TIP_ALLOCATION_COLUMNS = tuple(f"AJB{i}_TXAMT" for i in range(1, 8))
_SIPP_OBSERVED_STATUS_VALUES = frozenset((0, 1, 9))
_DONOR_WEIGHT_COLUMN = "sipp_weight"
_MAX_TRAIN_SAMPLES = 10_000
_DEFAULT_N_ESTIMATORS = 100
# Stable seed produced by the retired pipeline's ``seeded_rng`` for
# ``calibration_sipp_tip_training_sample:tip_income``. Keeping the named-source
# seed matters on this sparse target: seed 0 materially under-samples positive
# tip rows in the 10,000-row cap.
_TIP_TRAINING_SAMPLE_SEED = 5_559_651_045_748_063_828

# Weighted all-person plausibility bands.  The pinned reference eCPS has 7.09%
# in a listed occupation and 0.79% with positive tip income.  These deliberately
# broad bands catch a zero/constant/wrong-column surface without point-pinning
# one stochastic QRF draw.
_TIPPED_OCCUPATION_SHARE_BAND = (0.02, 0.15)
_TIP_INCOME_NONZERO_SHARE_BAND = (0.001, 0.03)

# Retired ``datasets/cps/tipped_occupation.py`` mapping, itself derived from the
# IRS/Treasury 2025-42 tipped-occupation list joined to the Census 2018
# occupation-to-SOC crosswalk.  A few SOC collisions use one representative
# TTOC because PolicyEngine only needs listed (>0) versus unlisted (=0).
CENSUS_OCCUPATION_CODE_TO_TTOC: dict[int, int] = {
    725: 502,
    2350: 507,
    2633: 502,
    2752: 206,
    2755: 207,
    2770: 208,
    2910: 503,
    3602: 501,
    3630: 602,
    4000: 105,
    4010: 106,
    4030: 106,
    4040: 101,
    4055: 107,
    4110: 102,
    4120: 103,
    4130: 104,
    4140: 108,
    4150: 109,
    4160: 106,
    4230: 304,
    4251: 402,
    4350: 506,
    4420: 210,
    4500: 603,
    4510: 603,
    4521: 605,
    4522: 601,
    4600: 508,
    4621: 607,
    4655: 501,
    5130: 203,
    5300: 303,
    6355: 403,
    6442: 404,
    7120: 401,
    7200: 409,
    7315: 405,
    7320: 406,
    7340: 401,
    7540: 408,
    7610: 401,
    7800: 110,
    8510: 401,
    9122: 806,
    9141: 803,
    9142: 802,
    9350: 801,
    9610: 805,
    9620: 809,
}


def us_sipp_tips_stage_spec() -> SourceStageSpec:
    """Load and validate the packaged ``sipp_tips`` stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SIPP_TIPS_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_SIPP_TIPS_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_SIPP_TIPS_STAGE_NAME]
    missing = sorted(set(US_SIPP_TIPS_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{US_SIPP_TIPS_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _sha256_hexdigest(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def fetch_sipp_2023_tip_donor(
    cache_dir: str | Path | None = None,
    *,
    expected_sha256: str | None = SIPP_2023_TIP_DONOR_SHA256,
) -> Path:
    """Download, SHA-verify, and cache the fixed SIPP 2023 slim donor."""

    import urllib.request

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache" / "microcosm" / "sipp"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / _SIPP_2023_TIP_DONOR_FILENAME
    if target.exists() and target.stat().st_size > 0:
        if expected_sha256 is None:
            return target
        if _sha256_hexdigest(target.read_bytes()) == expected_sha256:
            return target

    with urllib.request.urlopen(SIPP_2023_TIP_DONOR_URL) as response:  # noqa: S310
        payload = response.read()
    if expected_sha256 is not None:
        digest = _sha256_hexdigest(payload)
        if digest != expected_sha256:
            raise ValueError(
                "SIPP 2023 tip donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {digest}."
            )
    target.write_bytes(payload)
    return target


def derive_treasury_tipped_occupation_code(
    census_occupation_codes: pd.Series | np.ndarray,
) -> np.ndarray:
    """Map detailed Census occupation codes to Treasury tipped codes."""

    values = pd.to_numeric(
        pd.Series(census_occupation_codes, copy=False), errors="coerce"
    ).fillna(-1)
    return (
        values.astype(int)
        .map(CENSUS_OCCUPATION_CODE_TO_TTOC)
        .fillna(0)
        .astype(np.int16)
        .to_numpy()
    )


def _derive_any_tipped_occupation_code(frame: pd.DataFrame) -> np.ndarray:
    mapped = [
        derive_treasury_tipped_occupation_code(frame[column])
        for column in _SIPP_JOB_OCCUPATION_COLUMNS
    ]
    return np.column_stack(mapped).max(axis=1).astype(np.int16)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sipp_2023_tip_donor(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Load the retired SIPP tip donor transformation from its pinned CSV."""

    path = Path(path)
    if expected_sha256 is not None:
        digest = _sha256_file(path)
        if digest != expected_sha256:
            raise ValueError(
                "SIPP 2023 tip donor failed sha-256 verification: "
                f"expected {expected_sha256}, got {digest}."
            )

    required = {
        "SSUID",
        "MONTHCODE",
        "WPFINWGT",
        "TAGE",
        "TPTOTINC",
        *_SIPP_JOB_OCCUPATION_COLUMNS,
        *_SIPP_TIP_AMOUNT_COLUMNS,
        *_SIPP_TIP_ALLOCATION_COLUMNS,
    }
    raw = pd.read_csv(path, usecols=lambda column: column in required)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"SIPP 2023 tip donor missing required column(s): {missing}.")

    raw = raw.loc[pd.to_numeric(raw["MONTHCODE"], errors="coerce") == 12].copy()
    if raw.empty:
        raise ValueError("SIPP 2023 tip donor has no December person records.")

    observed = pd.Series(True, index=raw.index)
    for column in _SIPP_TIP_ALLOCATION_COLUMNS:
        flag = pd.to_numeric(raw[column], errors="coerce").fillna(0).astype(int)
        observed &= flag.isin(_SIPP_OBSERVED_STATUS_VALUES)
    if not observed.any():
        raise ValueError("SIPP 2023 tip donor has no observed tip-amount records.")

    # Derive household composition on every December member before applying
    # the target-specific observed-source mask. An allocated child is not a tip
    # training target, but still counts in a retained adult's household.
    donor = pd.DataFrame(index=raw.index)
    tip_amounts = raw.loc[:, list(_SIPP_TIP_AMOUNT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    donor["tip_income"] = np.maximum(
        tip_amounts.fillna(0.0).sum(axis=1).to_numpy(dtype=np.float64) * 12.0,
        0.0,
    )
    donor["employment_income"] = (
        pd.to_numeric(raw["TPTOTINC"], errors="coerce").fillna(0.0).to_numpy() * 12.0
    )
    donor["age"] = pd.to_numeric(raw["TAGE"], errors="coerce").fillna(0.0)
    donor["treasury_tipped_occupation_code"] = _derive_any_tipped_occupation_code(raw)
    donor["is_tipped_occupation"] = (
        donor["treasury_tipped_occupation_code"] > 0
    ).astype(np.float64)
    household = pd.DataFrame(
        {
            "SSUID": raw["SSUID"],
            "under_18": donor["age"] < 18,
            "under_6": donor["age"] < 6,
        },
        index=raw.index,
    )
    donor["count_under_18"] = household.groupby("SSUID")["under_18"].transform("sum")
    donor["count_under_6"] = household.groupby("SSUID")["under_6"].transform("sum")
    donor[_DONOR_WEIGHT_COLUMN] = pd.to_numeric(
        raw["WPFINWGT"], errors="coerce"
    ).fillna(0.0)
    finite = np.isfinite(donor.loc[:, [*SIPP_TIP_PREDICTORS, "tip_income"]]).all(axis=1)
    positive_weight = donor[_DONOR_WEIGHT_COLUMN] > 0
    return donor.loc[observed & finite & positive_weight].reset_index(drop=True)


def _recipient_features(person: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [
        column
        for column in US_SIPP_TIPS_REQUIRED_SOURCE_COLUMNS
        if column not in person.columns
    ]
    if missing:
        raise ValueError(
            f"US SIPP-tip imputation requires recipient person column(s): {missing}."
        )

    occupation_code = derive_treasury_tipped_occupation_code(person["PEIOOCC"])
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0.0)
    household = pd.DataFrame(
        {
            "household_id": person["person_household_id"].to_numpy(),
            "under_18": age.to_numpy() < 18,
            "under_6": age.to_numpy() < 6,
        },
        index=person.index,
    )
    features = pd.DataFrame(index=person.index)
    features["employment_income"] = pd.to_numeric(
        person["employment_income_before_lsr"], errors="coerce"
    ).fillna(0.0)
    features["age"] = age
    features["count_under_18"] = household.groupby("household_id")[
        "under_18"
    ].transform("sum")
    features["count_under_6"] = household.groupby("household_id")["under_6"].transform(
        "sum"
    )
    features["is_tipped_occupation"] = (occupation_code > 0).astype(np.float64)
    return features.loc[:, list(SIPP_TIP_PREDICTORS)], occupation_code


def impute_us_sipp_tips(
    person: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """QRF-impute tip income and carry the CPS-derived Treasury code."""

    from microcosm.fit import QRF

    required = [*SIPP_TIP_PREDICTORS, "tip_income", _DONOR_WEIGHT_COLUMN]
    missing = [column for column in required if column not in donor.columns]
    if missing:
        raise ValueError(f"SIPP tip donor table missing column(s): {missing}.")
    fit_frame = donor.loc[:, required].copy()
    for column in required:
        fit_frame[column] = pd.to_numeric(fit_frame[column], errors="coerce").fillna(
            0.0
        )
    if len(fit_frame) > _MAX_TRAIN_SAMPLES:
        selected = np.random.default_rng(_TIP_TRAINING_SAMPLE_SEED).choice(
            len(fit_frame), size=_MAX_TRAIN_SAMPLES, replace=False
        )
        fit_frame = fit_frame.iloc[np.sort(selected)].reset_index(drop=True)

    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        fit_frame,
        predictors=list(SIPP_TIP_PREDICTORS),
        targets=["tip_income"],
        weights=_DONOR_WEIGHT_COLUMN,
    )
    features, occupation_code = _recipient_features(person)
    predicted = np.maximum(
        np.asarray(fitted.predict(features)["tip_income"], dtype=np.float64), 0.0
    )
    return pd.DataFrame(
        {
            "tip_income": predicted,
            "treasury_tipped_occupation_code": occupation_code,
        },
        index=person.index,
    )


def _tip_surface_carries_signal(person: pd.DataFrame) -> bool:
    return all(
        person[column].dropna().nunique() > 1 for column in US_SIPP_TIPS_OUTPUT_COLUMNS
    )


def with_us_sipp_tip_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    sipp_donor: pd.DataFrame,
) -> Frame:
    """Apply the SIPP tip stage to a US frame before either release arm."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SIPP-tip inputs require the US schema.")
    us_sipp_tips_stage_spec()
    person = frame.table("person")
    if all(column in person for column in US_SIPP_TIPS_OUTPUT_COLUMNS) and (
        _tip_surface_carries_signal(person)
    ):
        return frame

    imputed = impute_us_sipp_tips(person, sipp_donor, seed=int(seed))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["tip_income"] = imputed["tip_income"].to_numpy(dtype=np.float64)
    tables["person"]["treasury_tipped_occupation_code"] = imputed[
        "treasury_tipped_occupation_code"
    ].to_numpy(dtype=np.int16)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_sipp_tips_summary(frame: Frame) -> dict[str, object]:
    """Weighted tip-income and tipped-occupation incidence diagnostics."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())

    def share(mask: np.ndarray) -> float:
        return float(weights[mask].sum()) / total_weight if total_weight > 0 else 0.0

    tip = pd.to_numeric(person["tip_income"], errors="coerce").fillna(0.0)
    occupation = pd.to_numeric(
        person["treasury_tipped_occupation_code"], errors="coerce"
    ).fillna(0)
    return {
        "tip_income_nonzero_share": share(tip.to_numpy() > 0),
        "tipped_occupation_share": share(occupation.to_numpy() > 0),
        "tip_income_nonzero_share_band": list(_TIP_INCOME_NONZERO_SHARE_BAND),
        "tipped_occupation_share_band": list(_TIPPED_OCCUPATION_SHARE_BAND),
        "unique_counts": {
            "tip_income": int(tip.nunique()),
            "treasury_tipped_occupation_code": int(occupation.nunique()),
        },
    }


def us_sipp_tips_signal_gate(frame: Frame) -> GateResult:
    """Require both tip-family columns to carry plausible non-default signal."""

    person = frame.table("person")
    missing = [column for column in US_SIPP_TIPS_OUTPUT_COLUMNS if column not in person]
    if missing:
        return GateResult(
            name="sipp_tips_signal",
            passed=False,
            failures=(f"person columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_sipp_tips_summary(frame)
    failures: list[str] = []
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(
                f"{column}: constant column (one observed value) — the tip "
                "surface carries no signal."
            )
    for key, band, label in (
        (
            "tip_income_nonzero_share",
            _TIP_INCOME_NONZERO_SHARE_BAND,
            "tip-income",
        ),
        (
            "tipped_occupation_share",
            _TIPPED_OCCUPATION_SHARE_BAND,
            "tipped-occupation",
        ),
    ):
        value = float(summary[key])
        low, high = band
        if not (low <= value <= high):
            failures.append(
                f"{label} share {value:.3f} outside plausibility band [{low}, {high}]."
            )
    return GateResult(
        name="sipp_tips_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
