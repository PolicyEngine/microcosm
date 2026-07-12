"""Current-vintage SPI QRF stages for the UK national support channel."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import FitWeightRecord
from populace.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
    SPI_INCOME_IMPUTATION_COLUMNS,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UKSPISupportResult,
    support_channel_column,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

QRF: Any | None = None

SPI_DONOR_RELEASE = "spi_2022_23"
SPI_DONOR_FILENAME = "put2223uk.tab"
SPI_DONOR_VINTAGE = "2022-23"
SPI_DONOR_UKDS_STUDY = "SN 9422"
SPI_DONOR_DOI = "10.5255/UKDA-SN-9422-1"
SPI_DONOR_FIT_NAME = "uk_spi_2022_23_income"
FRS_ONLY_FIT_NAME = "uk_frs_only_spi_fill"
DEFAULT_SPI_DONOR_SAMPLE_SIZE = 100_000
SPI_TI_IDENTITY_ABS_TOLERANCE_GBP = 100.0

SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS = {
    "incapacity_benefit_reported": (
        "Absent/all-default on the pinned enhanced-FRS export and certified "
        "Populace UK base; not a populated loader layer."
    ),
    "maternity_allowance_reported": (
        "Absent from the pinned enhanced-FRS export and certified Populace UK "
        "base; no training source can be materialized for this stage."
    ),
}

SPI_DONOR_REQUIRED_COLUMNS = (
    "SEX",
    "FACT",
    "GORCODE",
    "AGERANGE",
    "PAY",
    "EPB",
    "EXPS",
    "TAXTERM",
    "INCPBEN",
    "OSSBEN",
    "UBISJA",
    "MOTHINC",
    "OTHERINC",
    "PROFITS",
    "CAPALL",
    "LOSSBF",
    "SRP",
    "INCBBS",
    "DIVIDENDS",
    "PENSION",
    "INCPROP",
    "OTHERINV",
    "GIFTAID",
    "GIFTINV",
    "TI",
)
SPI_INCOME_SOURCE_COLUMNS = {
    "employment_income": (
        "PAY",
        "EPB",
        "EXPS",
        "INCPBEN",
        "OSSBEN",
        "TAXTERM",
        "UBISJA",
        "MOTHINC",
    ),
    "self_employment_income": ("PROFITS", "CAPALL", "LOSSBF"),
    "savings_interest_income": ("INCBBS",),
    "dividend_income": ("DIVIDENDS",),
    "private_pension_income": ("PENSION",),
    "property_income": ("INCPROP",),
    "other_investment_income": ("OTHERINV",),
    "gift_aid": ("GIFTAID",),
    "charitable_investment_gifts": ("GIFTINV",),
    "hmrc_spi_assessable_income": ("TI",),
}
_DIRECT_SPI_OUTPUT_SOURCE_COLUMNS = {
    output: sources
    for output, sources in SPI_INCOME_SOURCE_COLUMNS.items()
    if output not in {"employment_income", "self_employment_income"}
}
_SPI_AGE_RANGES = {
    -1: (16, 70),
    1: (16, 25),
    2: (25, 35),
    3: (35, 45),
    4: (45, 55),
    5: (55, 65),
    6: (65, 74),
    7: (74, 90),
}
_SPI_REGION_MAP = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}


@dataclass(frozen=True)
class UKSPIIncomeImputationResult:
    """SPI-filled person table and auditable fit/source evidence."""

    person: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]
    donor_path: Path
    donor_sha256: str
    donor_rows: int
    stage2_training_rows: int
    spi_prediction_rows: int
    reviewed_absent_stage2_outputs: dict[str, str]


def impute_uk_spi_income_support(
    support: UKSPISupportResult,
    spi_tab_path: str | Path,
    *,
    seed: int = 42,
    n_estimators: int = 100,
    donor_sample_size: int | None = DEFAULT_SPI_DONOR_SAMPLE_SIZE,
    build_period: int | str = 2023,
) -> UKSPIIncomeImputationResult:
    """Run strict SPI-income and FRS-only QRFs on rebuilt positive support."""

    if support.household_weight_kind is not WeightKind.IMPORTANCE:
        raise ValueError(
            "SPI income imputation requires rebuilt importance-weight support."
        )
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not isinstance(n_estimators, int) or n_estimators <= 0:
        raise ValueError("n_estimators must be a positive integer.")
    if donor_sample_size is not None and (
        not isinstance(donor_sample_size, int) or donor_sample_size <= 0
    ):
        raise ValueError("donor_sample_size must be a positive integer or None.")

    donor_path = Path(spi_tab_path).expanduser().resolve()
    if donor_path.name != SPI_DONOR_FILENAME:
        raise ValueError(
            f"Current SPI donor must be named {SPI_DONOR_FILENAME!r}, got "
            f"{donor_path.name!r}."
        )
    if not donor_path.is_file():
        raise FileNotFoundError(f"SPI 2022-23 donor not found: {donor_path}.")
    raw_donor = pd.read_csv(donor_path, delimiter="\t")
    donor = _prepare_spi_donor(raw_donor, seed=seed)
    donor_fit_weights = donor["FACT"].to_numpy(dtype=np.float64)
    if donor_sample_size is not None:
        donor = donor.sample(
            n=donor_sample_size,
            replace=True,
            weights="FACT",
            random_state=seed,
        ).reset_index(drop=True)
        # Mirroring the enhanced-FRS pipeline, the FACT-weighted bootstrap
        # itself represents the SPI design. Reapplying FACT to the sampled
        # rows would square the survey weights. Uniform typed DESIGN weights
        # keep the fit auditable without double-weighting the donor.
        donor_fit_weights = np.ones(len(donor), dtype=np.float64)

    person = support.person.copy()
    household = support.household
    person_channel = support_channel_column("person")
    household_channel = support_channel_column("household")
    _require_columns(person, (person_channel,), label="person support")
    _require_columns(
        household,
        (
            "household_id",
            "household_weight",
            household_channel,
            HOUSEHOLD_IS_SPI_SYNTHETIC_COLUMN,
        ),
        label="household support",
    )
    spi_household = household[household_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if (
        not spi_household.any()
        or not household.loc[spi_household, "household_weight"].gt(0.0).all()
    ):
        raise ValueError("SPI support rows must all carry positive effective mass.")
    spi_people = person[person_channel] == SPI_SYNTHETIC_SUPPORT_CHANNEL
    if not spi_people.any():
        raise ValueError("SPI support has no person rows to impute.")

    recipient_predictors = _person_predictors(
        person.loc[spi_people],
        household,
        income_predictors=(),
    )
    donor_predictors, encoded_recipient = _encode_predictor_pair(
        donor[["age", "gender", "region"]],
        recipient_predictors,
    )
    donor_frame = _person_fit_frame(
        predictors=donor_predictors,
        targets=donor[list(SPI_INCOME_QRF_OUTPUT_COLUMNS)],
        weights=donor_fit_weights,
        weight_kind=WeightKind.DESIGN,
    )
    qrf_cls = _qrf_class()
    stage1 = qrf_cls(n_estimators=n_estimators, seed=seed).fit(
        donor_frame,
        list(donor_predictors.columns),
        list(SPI_INCOME_QRF_OUTPUT_COLUMNS),
        weights="design",
    )
    stage1_draws = stage1.predict(encoded_recipient)
    _validate_predictions(
        stage1_draws,
        expected=SPI_INCOME_QRF_OUTPUT_COLUMNS,
        label="SPI stage-1",
    )
    for column in SPI_INCOME_QRF_OUTPUT_COLUMNS:
        if column not in person:
            person[column] = 0.0
        person.loc[spi_people, column] = stage1_draws[column].to_numpy()

    taxable_interest_draw = person.loc[spi_people, "savings_interest_income"].to_numpy(
        dtype=np.float64, copy=True
    )
    stage2_outputs, reviewed_absent = _stage2_outputs(person)
    base_household = household[household_channel] == BASE_FRS_SUPPORT_CHANNEL
    if "clone_index" in household:
        base_household &= household["clone_index"] == 0
    training_household_ids = set(household.loc[base_household, "household_id"])
    training_people = (person[person_channel] == BASE_FRS_SUPPORT_CHANNEL) & person[
        "person_household_id"
    ].isin(training_household_ids)
    if not training_people.any():
        raise ValueError("FRS-only stage has no canonical base training rows.")

    income_predictors = tuple(
        column
        for column in SPI_INCOME_IMPUTATION_COLUMNS
        if column not in {"gift_aid", "charitable_investment_gifts"}
    )
    train_predictors = _person_predictors(
        person.loc[training_people],
        household,
        income_predictors=income_predictors,
    )
    target_predictors = _person_predictors(
        person.loc[spi_people],
        household,
        income_predictors=income_predictors,
    )
    encoded_train, encoded_target = _encode_predictor_pair(
        train_predictors,
        target_predictors,
    )
    training_targets = person.loc[training_people, list(stage2_outputs)].copy()
    _require_finite_numeric(training_targets, label="FRS-only training outputs")
    if (training_targets.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("FRS-only training outputs must be non-negative.")
    person_weights = _person_household_weights(person, household)
    stage2_frame = _person_fit_frame(
        predictors=encoded_train,
        targets=training_targets,
        weights=person_weights.loc[training_people].to_numpy(dtype=np.float64),
        weight_kind=WeightKind.IMPORTANCE,
    )
    stage2 = qrf_cls(n_estimators=n_estimators, seed=seed + 1).fit(
        stage2_frame,
        list(encoded_train.columns),
        list(stage2_outputs),
        weights="importance",
    )
    stage2_draws = stage2.predict(encoded_target)
    _validate_predictions(
        stage2_draws,
        expected=stage2_outputs,
        label="FRS-only stage-2",
    )
    if (stage2_draws.to_numpy(dtype=np.float64) < 0.0).any():
        raise ValueError("FRS-only stage-2 produced negative non-negative outputs.")
    for column in stage2_outputs:
        person.loc[spi_people, column] = stage2_draws[column].to_numpy()

    tax_free = person.loc[spi_people, "tax_free_savings_income"].to_numpy(
        dtype=np.float64
    )
    person.loc[spi_people, "savings_interest_income"] = taxable_interest_draw + tax_free
    person = _refresh_disability_derived_inputs(
        person,
        spi_people=spi_people,
        build_period=build_period,
    )
    return UKSPIIncomeImputationResult(
        person=person,
        fit_weight_records=(
            FitWeightRecord(SPI_DONOR_FIT_NAME, stage1.weight_kind),
            FitWeightRecord(FRS_ONLY_FIT_NAME, stage2.weight_kind),
        ),
        donor_path=donor_path,
        donor_sha256=_sha256(donor_path),
        donor_rows=len(donor),
        stage2_training_rows=int(training_people.sum()),
        spi_prediction_rows=int(spi_people.sum()),
        reviewed_absent_stage2_outputs=reviewed_absent,
    )


def _prepare_spi_donor(raw: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    _require_columns(raw, SPI_DONOR_REQUIRED_COLUMNS, label="SPI 2022-23 donor")
    numeric = pd.DataFrame(
        {
            column: pd.to_numeric(raw[column], errors="coerce")
            for column in SPI_DONOR_REQUIRED_COLUMNS
        }
    )
    _require_finite_numeric(numeric, label="SPI 2022-23 donor")
    if not (numeric["FACT"] > 0.0).all():
        raise ValueError("SPI 2022-23 FACT weights must be strictly positive.")
    sex = numeric["SEX"]
    if not sex.isin([0, 1, 2]).all():
        raise ValueError(
            "SPI 2022-23 SEX must contain only documented codes 0/1/2."
        )
    age_codes = numeric["AGERANGE"].astype(int)
    unknown_age = sorted(set(age_codes) - set(_SPI_AGE_RANGES))
    if unknown_age:
        raise ValueError(f"SPI 2022-23 has unknown AGERANGE code(s): {unknown_age}.")
    rng = np.random.default_rng(seed)
    bounds = np.asarray([_SPI_AGE_RANGES[code] for code in age_codes])
    donor = pd.DataFrame(
        {
            "age": bounds[:, 0] + rng.random(len(raw)) * (bounds[:, 1] - bounds[:, 0]),
            "gender": np.select(
                (sex == 1, sex == 2),
                ("MALE", "FEMALE"),
                default="UNKNOWN",
            ),
            "region": numeric["GORCODE"]
            .astype(int)
            .map(_SPI_REGION_MAP)
            .fillna("UNKNOWN"),
            "FACT": numeric["FACT"],
            "employment_income": (
                np.maximum(
                    numeric["PAY"] + numeric["EPB"] - numeric["EXPS"],
                    0.0,
                )
                + numeric["INCPBEN"]
                + numeric["OSSBEN"]
                + numeric["TAXTERM"]
                + numeric["UBISJA"]
                + numeric["MOTHINC"]
            ),
            "self_employment_income": np.maximum(
                numeric["PROFITS"] - numeric["CAPALL"] - numeric["LOSSBF"],
                0.0,
            ),
        }
    )
    for output, sources in _DIRECT_SPI_OUTPUT_SOURCE_COLUMNS.items():
        donor[output] = numeric[list(sources)].sum(axis=1)
    _require_finite_numeric(
        donor[["age", "FACT", *SPI_INCOME_QRF_OUTPUT_COLUMNS]],
        label="SPI 2022-23 derived donor",
    )
    derived_ti = (
        donor["employment_income"]
        + numeric["OTHERINC"]
        + numeric["SRP"]
        + numeric["PENSION"]
        + donor["self_employment_income"]
        + numeric["OTHERINV"]
        + numeric["DIVIDENDS"]
        + numeric["INCPROP"]
        + numeric["INCBBS"]
    )
    ti_error = np.abs(donor["hmrc_spi_assessable_income"] - derived_ti)
    if (ti_error > SPI_TI_IDENTITY_ABS_TOLERANCE_GBP).any():
        worst = float(ti_error.max())
        raise ValueError(
            "SPI 2022-23 TI disagrees with the published TEI + TII identity; "
            f"worst absolute difference {worst:.6g} exceeds the reviewed "
            f"£{SPI_TI_IDENTITY_ABS_TOLERANCE_GBP:.0f} rounding tolerance."
        )
    for column in ("gift_aid", "charitable_investment_gifts"):
        if (donor[column] < 0.0).any():
            raise ValueError(f"SPI donor {column} must be non-negative.")
    return donor


def _person_predictors(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    income_predictors: tuple[str, ...],
) -> pd.DataFrame:
    required = ("person_household_id", "age", "gender", *income_predictors)
    _require_columns(person, required, label="SPI QRF person predictors")
    _require_columns(household, ("household_id", "region"), label="household")
    region = household.set_index("household_id")["region"]
    mapped_region = person["person_household_id"].map(region)
    if mapped_region.isna().any():
        raise ValueError("SPI QRF cannot map every person to a household region.")
    result = person[["age", "gender", *income_predictors]].copy()
    result["region"] = mapped_region.to_numpy()
    result = result[["age", "gender", "region", *income_predictors]]
    _require_finite_numeric(
        result[["age", *income_predictors]],
        label="SPI QRF numeric predictors",
    )
    if result[["gender", "region"]].isna().any().any():
        raise ValueError("SPI QRF categorical predictors contain missing values.")
    return result


def _encode_predictor_pair(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train.reset_index(drop=True), target.reset_index(drop=True)],
        ignore_index=True,
    )
    encoded = pd.get_dummies(
        combined,
        columns=["gender", "region"],
        drop_first=False,
        dtype=float,
    )
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)
    train_encoded = encoded.iloc[: len(train)].copy()
    train_encoded.index = train.index
    target_encoded = encoded.iloc[len(train) :].copy()
    target_encoded.index = target.index
    _require_finite_numeric(train_encoded, label="encoded QRF training predictors")
    _require_finite_numeric(target_encoded, label="encoded QRF target predictors")
    return train_encoded, target_encoded


def _person_fit_frame(
    *,
    predictors: pd.DataFrame,
    targets: pd.DataFrame,
    weights: np.ndarray,
    weight_kind: WeightKind,
) -> Frame:
    if len(predictors) != len(targets) or len(predictors) != len(weights):
        raise ValueError("QRF predictors, targets, and weights must align.")
    ids = np.arange(1, len(predictors) + 1, dtype=np.int64)
    person = pd.concat(
        [predictors.reset_index(drop=True), targets.reset_index(drop=True)],
        axis=1,
    )
    person.insert(0, "person_household_id", ids)
    person.insert(0, "person_id", ids)
    household = pd.DataFrame({"household_id": ids})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights), weight_kind)},
    )


def _stage2_outputs(person: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, str]]:
    outputs: list[str] = []
    reviewed: dict[str, str] = {}
    missing_unreviewed: list[str] = []
    for column in FRS_ONLY_SPI_FILL_PERSON_COLUMNS:
        if column in SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS:
            if column in person:
                values = pd.to_numeric(person[column], errors="coerce").to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Reviewed-absent FRS-only output {column!r} contains "
                        "non-finite values."
                    )
                if (values != 0.0).any():
                    raise ValueError(
                        f"Reviewed-absent FRS-only output {column!r} now carries "
                        "non-default source signal; update and review the source "
                        "manifest before adding it to the QRF surface."
                    )
            reviewed[column] = SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS[column]
        elif column in person:
            outputs.append(column)
        else:
            missing_unreviewed.append(column)
    if missing_unreviewed:
        raise ValueError(
            "FRS-only stage cannot silently narrow missing output(s): "
            f"{missing_unreviewed}."
        )
    if not outputs:
        raise ValueError("FRS-only stage has no materializable outputs.")
    return tuple(outputs), reviewed


def _person_household_weights(
    person: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.Series:
    weights = household.set_index("household_id")["household_weight"]
    mapped = person["person_household_id"].map(weights)
    if mapped.isna().any() or not np.isfinite(mapped.to_numpy(dtype=float)).all():
        raise ValueError("Cannot resolve finite household weights for every person.")
    if not (mapped > 0.0).any():
        raise ValueError("Resolved person weights contain no positive mass.")
    return mapped.astype(float)


def _validate_predictions(
    predictions: pd.DataFrame,
    *,
    expected: tuple[str, ...],
    label: str,
) -> None:
    if tuple(predictions.columns) != tuple(expected):
        raise ValueError(
            f"{label} prediction surface mismatch: expected {list(expected)}, "
            f"got {list(predictions.columns)}."
        )
    _require_finite_numeric(predictions, label=f"{label} predictions")


def _require_finite_numeric(frame: pd.DataFrame, *, label: str) -> None:
    numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain only finite numeric values.")


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing column(s): {missing}.")


def _qrf_class():
    if QRF is not None:
        return QRF
    return import_module("populace.fit").QRF


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def _disability_parameters(year: int):
    from policyengine_uk import CountryTaxBenefitSystem
    from policyengine_uk.model_api import WEEKS_IN_YEAR

    system = CountryTaxBenefitSystem()
    return (
        system.parameters(year).baseline.gov.dwp,
        system.parameters(year).gov.dwp,
        float(WEEKS_IN_YEAR),
    )


def _refresh_disability_derived_inputs(
    person: pd.DataFrame,
    *,
    spi_people: pd.Series,
    build_period: int | str,
) -> pd.DataFrame:
    """Keep category/flag inputs coherent with stage-2 reported amounts."""

    try:
        year = int(str(build_period)[:4])
    except ValueError as exc:
        raise ValueError(f"Invalid UK SPI build period {build_period!r}.") from exc
    baseline_dwp, dwp, weeks_in_year = _disability_parameters(year)
    target = person.loc[spi_people].copy()
    mappings = (
        (
            "attendance_allowance_reported",
            "aa_category",
            (
                ("LOWER", baseline_dwp.attendance_allowance.lower),
                ("HIGHER", baseline_dwp.attendance_allowance.higher),
            ),
        ),
        (
            "dla_sc_reported",
            "dla_sc_category",
            (
                ("LOWER", baseline_dwp.dla.self_care.lower),
                ("MIDDLE", baseline_dwp.dla.self_care.middle),
                ("HIGHER", baseline_dwp.dla.self_care.higher),
            ),
        ),
        (
            "dla_m_reported",
            "dla_m_category",
            (
                ("LOWER", baseline_dwp.dla.mobility.lower),
                ("HIGHER", baseline_dwp.dla.mobility.higher),
            ),
        ),
        (
            "pip_m_reported",
            "pip_m_category",
            (
                ("STANDARD", baseline_dwp.pip.mobility.standard),
                ("ENHANCED", baseline_dwp.pip.mobility.enhanced),
            ),
        ),
        (
            "pip_dl_reported",
            "pip_dl_category",
            (
                ("STANDARD", baseline_dwp.pip.daily_living.standard),
                ("ENHANCED", baseline_dwp.pip.daily_living.enhanced),
            ),
        ),
    )
    for reported, category, thresholds in mappings:
        if reported not in target:
            continue
        weekly = pd.to_numeric(target[reported], errors="coerce").fillna(0.0)
        weekly = weekly.to_numpy(dtype=float) / weeks_in_year
        values = np.full(len(target), "NONE", dtype=object)
        for name, rate in thresholds:
            threshold = max(0.0, float(rate) - 1.0)
            values[weekly >= threshold] = name
        _assign_spi_values(person, spi_people, category, values, default="NONE")

    reported_flag_columns = (
        "attendance_allowance_reported",
        "dla_sc_reported",
        "dla_m_reported",
        "pip_m_reported",
        "pip_dl_reported",
        "sda_reported",
        "incapacity_benefit_reported",
        "iidb_reported",
        "afcs_reported",
        "esa_contrib_reported",
        "esa_income_reported",
    )
    total = np.zeros(len(target), dtype=float)
    for column in reported_flag_columns:
        if column in target:
            total += pd.to_numeric(target[column], errors="coerce").fillna(0.0)
    _assign_spi_values(
        person,
        spi_people,
        "is_disabled_for_benefits",
        total > 0.0,
        default=False,
    )

    def amount(column: str) -> np.ndarray:
        if column not in target:
            return np.zeros(len(target), dtype=float)
        return pd.to_numeric(target[column], errors="coerce").fillna(0.0).to_numpy()

    annual_weeks = 365.25 / 7.0
    safety_gap = annual_weeks
    attendance = amount("attendance_allowance_reported")
    dla_sc = amount("dla_sc_reported")
    pip_dl = amount("pip_dl_reported")
    afcs = amount("afcs_reported")
    aa_higher = float(dwp.attendance_allowance.higher) * annual_weeks - safety_gap
    dla_higher = float(dwp.dla.self_care.higher) * annual_weeks - safety_gap
    pip_enhanced = float(dwp.pip.daily_living.enhanced) * annual_weeks - safety_gap
    _assign_spi_values(
        person,
        spi_people,
        "is_enhanced_disabled_for_benefits",
        (attendance >= aa_higher) | (dla_sc > dla_higher) | (pip_dl >= pip_enhanced),
        default=False,
    )
    _assign_spi_values(
        person,
        spi_people,
        "is_severely_disabled_for_benefits",
        (attendance > 0.0)
        | (dla_sc >= dla_higher)
        | (pip_dl >= pip_enhanced)
        | (afcs > 0.0),
        default=False,
    )
    return person


def _assign_spi_values(
    person: pd.DataFrame,
    spi_people: pd.Series,
    column: str,
    values: np.ndarray,
    *,
    default: object,
) -> None:
    if column not in person:
        person[column] = default
    elif isinstance(person[column].dtype, pd.CategoricalDtype):
        person[column] = person[column].astype(object)
    person.loc[spi_people, column] = values


__all__ = [
    "DEFAULT_SPI_DONOR_SAMPLE_SIZE",
    "FRS_ONLY_FIT_NAME",
    "SPI_DONOR_DOI",
    "SPI_DONOR_FILENAME",
    "SPI_DONOR_FIT_NAME",
    "SPI_DONOR_RELEASE",
    "SPI_DONOR_UKDS_STUDY",
    "SPI_DONOR_VINTAGE",
    "SPI_DONOR_REQUIRED_COLUMNS",
    "SPI_INCOME_SOURCE_COLUMNS",
    "SPI_STAGE2_REVIEWED_ABSENT_OUTPUTS",
    "SPI_TI_IDENTITY_ABS_TOLERANCE_GBP",
    "UKSPIIncomeImputationResult",
    "impute_uk_spi_income_support",
]
