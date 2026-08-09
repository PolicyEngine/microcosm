"""Household auto-loan inputs restored from the full 2022 SCF.

The public SCF summary extract used by :mod:`scf_wealth` contains the common
demographic/income predictors but not the four vehicle-loan amount/rate pairs.
Those live in the full public extract.  This module joins the two artifacts on
``(y1, yy1)`` (family and implicate), derives the retired eCPS targets, and fits
one weighted multi-target QRF draw per recipient household.

The two legacy outputs preserve the retired implementation's semantics:
``auto_loan_balance`` is the sum of the four *original financed amounts*, not
the remaining principal balance, and ``auto_loan_interest`` is each amount
times its reported rate.  Negative SCF codes are floored before rates are
divided by 10,000.

OBBBA uses a distinct later pure input,
``qualified_passenger_vehicle_loan_interest``.  SCF 2022 cannot observe a loan
originated after 2024 or final-assembly eligibility.  Treasury and IRS estimate
roughly six million qualifying loans are issued annually (16m new light-vehicle
sales × 60% financed × 60% final assembly in the United States).  The port uses
that official annual incidence as an expected-share proxy: six million divided
by weighted households with positive imputed interest, capped at one, times
each household's interest.  This is deliberately transparent and conservative;
it restores a nonzero, correctly bounded reform input without pretending the
2022 donor directly observed a post-2024 statutory fact.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec, load_source_manifest
from microcosm.build.us_runtime.eligibility_inputs import _own_children_in_household
from microcosm.build.us_runtime.scf_wealth import (
    SCF_WEALTH_PREDICTORS,
    _recipient_cps_race,
    _scf_summary_predictor_table,
    _sum_present,
)
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET",
    "SCF_2022_FULL_EXTRACT_MEMBER",
    "SCF_2022_FULL_EXTRACT_MEMBER_SHA256",
    "SCF_2022_FULL_EXTRACT_URL",
    "SCF_2022_FULL_EXTRACT_ZIP_SHA256",
    "SCF_AUTO_LOAN_AMOUNT_COLUMNS",
    "SCF_AUTO_LOAN_RATE_COLUMNS",
    "US_SCF_AUTO_LOAN_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_SCF_AUTO_LOAN_OUTPUT_COLUMNS",
    "fetch_scf_2022_full_extract",
    "impute_us_scf_auto_loans",
    "load_scf_2022_auto_loan_donor",
    "qualified_auto_loan_interest_proxy",
    "us_scf_auto_loans_signal_gate",
    "us_scf_auto_loans_stage_spec",
    "us_scf_auto_loans_summary",
    "with_us_scf_auto_loan_inputs",
]

SCF_2022_FULL_EXTRACT_URL = "https://www.federalreserve.gov/econres/files/scf2022s.zip"
SCF_2022_FULL_EXTRACT_MEMBER = "p22i6.dta"

# The source/member coordinates and 22,975-row public-file shape were verified
# against the Federal Reserve SCF download page and a successful retired build.
# The full archive itself was not retained locally.  Keep these explicit rather
# than (incorrectly) reusing the unrelated summary-extract hashes; one
# network-enabled provisioning run can fill the two pins without changing any
# transformation semantics.
SCF_2022_FULL_EXTRACT_ZIP_SHA256: str | None = None
SCF_2022_FULL_EXTRACT_MEMBER_SHA256: str | None = None

SCF_AUTO_LOAN_AMOUNT_COLUMNS: tuple[str, ...] = (
    "x2209",
    "x2309",
    "x2409",
    "x7158",
)
SCF_AUTO_LOAN_RATE_COLUMNS: tuple[str, ...] = (
    "x2219",
    "x2319",
    "x2419",
    "x7170",
)

US_SCF_AUTO_LOAN_OUTPUT_COLUMNS: tuple[str, ...] = (
    "auto_loan_balance",
    "auto_loan_interest",
    "qualified_passenger_vehicle_loan_interest",
)
US_SCF_AUTO_LOAN_NONCONSTANT_HOUSEHOLD_COLUMNS = US_SCF_AUTO_LOAN_OUTPUT_COLUMNS

QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET = 6_000_000.0

_SCF_STAGE_NAME = "scf_wealth"
_DONOR_WEIGHT_COLUMN = "scf_weight"
_HOUSEHOLD_ID_COLUMN = "person_household_id"
_DEFAULT_N_ESTIMATORS = 100
_AUTO_LOAN_NONZERO_SHARE_BAND = (0.10, 0.60)
_SCF_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_RECIPIENT_INTEREST_DIVIDEND_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
_RECIPIENT_SS_PENSION_COLUMNS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
)
_REQUIRED_RECIPIENT_PERSON_COLUMNS = (
    _HOUSEHOLD_ID_COLUMN,
    "age",
    "is_female",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "A_LINENO",
    "PH_SEQ",
    "PEPAR1",
    "PEPAR2",
    "employment_income_before_lsr",
)


def _sha256_hexdigest(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def us_scf_auto_loans_stage_spec() -> SourceStageSpec:
    """Load and validate the shared ``scf_wealth`` stage declaration."""

    manifest = load_source_manifest(
        files("microcosm.build.us").joinpath("source_stages.json")
    )
    spec = manifest.stage_map()[_SCF_STAGE_NAME]
    missing = sorted(set(US_SCF_AUTO_LOAN_OUTPUT_COLUMNS) - set(spec.outputs))
    if missing:
        raise ValueError(
            f"{_SCF_STAGE_NAME!r} manifest stage does not declare auto-loan "
            f"output(s) {missing}."
        )
    return spec


def fetch_scf_2022_full_extract(
    cache_dir: str | Path | None = None,
    *,
    expected_member_sha256: str | None = SCF_2022_FULL_EXTRACT_MEMBER_SHA256,
    expected_zip_sha256: str | None = SCF_2022_FULL_EXTRACT_ZIP_SHA256,
) -> Path:
    """Download, optionally verify, extract, and cache the full 2022 SCF."""

    import io
    import urllib.request
    import zipfile

    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "microcosm" / "scf"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / SCF_2022_FULL_EXTRACT_MEMBER
    if target.exists() and target.stat().st_size > 0:
        if expected_member_sha256 is None:
            return target
        digest = _sha256_hexdigest(target.read_bytes())
        if digest == expected_member_sha256:
            return target

    request = urllib.request.Request(
        SCF_2022_FULL_EXTRACT_URL,
        headers={"User-Agent": _SCF_HTTP_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        payload = response.read()
    if expected_zip_sha256 is not None:
        digest = _sha256_hexdigest(payload)
        if digest != expected_zip_sha256:
            raise ValueError(
                "SCF 2022 full-extract zip failed sha-256 verification: "
                f"expected {expected_zip_sha256}, got {digest}."
            )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if SCF_2022_FULL_EXTRACT_MEMBER not in archive.namelist():
            raise ValueError(
                "SCF 2022 full-extract zip is missing expected member "
                f"{SCF_2022_FULL_EXTRACT_MEMBER!r}."
            )
        member_bytes = archive.read(SCF_2022_FULL_EXTRACT_MEMBER)
    if expected_member_sha256 is not None:
        digest = _sha256_hexdigest(member_bytes)
        if digest != expected_member_sha256:
            raise ValueError(
                "SCF 2022 full-extract member failed sha-256 verification: "
                f"expected {expected_member_sha256}, got {digest}."
            )
    target.write_bytes(member_bytes)
    return target


def load_scf_2022_auto_loan_donor(
    summary_extract_path: str | Path,
    full_extract_path: str | Path,
) -> pd.DataFrame:
    """Join SCF summary/full implicates and derive the two retired targets."""

    join_columns = ["y1", "yy1"]
    summary = pd.read_stata(summary_extract_path, convert_categoricals=False)
    try:
        full = pd.read_stata(
            full_extract_path,
            convert_categoricals=False,
            columns=[
                *join_columns,
                *SCF_AUTO_LOAN_AMOUNT_COLUMNS,
                *SCF_AUTO_LOAN_RATE_COLUMNS,
            ],
        )
    except ValueError as error:
        if "not found in the Stata data set" not in str(error):
            raise
        raise ValueError(
            f"SCF 2022 full extract missing required column(s): {error}."
        ) from error
    for label, table, required in (
        (
            "summary",
            summary,
            {*join_columns},
        ),
        (
            "full",
            full,
            {
                *join_columns,
                *SCF_AUTO_LOAN_AMOUNT_COLUMNS,
                *SCF_AUTO_LOAN_RATE_COLUMNS,
            },
        ),
    ):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(
                f"SCF 2022 {label} extract missing required column(s): {missing}."
            )
        if table.duplicated(join_columns).any():
            raise ValueError(
                f"SCF 2022 {label} extract does not have one-to-one (y1, yy1) keys."
            )

    predictors = _scf_summary_predictor_table(summary)
    keyed_summary = summary.loc[:, join_columns].reset_index(drop=True)
    keyed_summary["_summary_order"] = np.arange(len(keyed_summary))
    keyed_full = full.loc[
        :,
        [
            *join_columns,
            *SCF_AUTO_LOAN_AMOUNT_COLUMNS,
            *SCF_AUTO_LOAN_RATE_COLUMNS,
        ],
    ]
    joined = keyed_summary.merge(
        keyed_full,
        on=join_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched = joined.loc[joined["_merge"] != "both", join_columns + ["_merge"]]
    if not unmatched.empty:
        raise ValueError(
            "SCF 2022 summary/full extracts have unmatched (y1, yy1) implicates; "
            f"a one-to-one join is required ({len(unmatched)} unmatched row(s))."
        )
    joined = joined.sort_values("_summary_order").reset_index(drop=True)

    amounts = joined.loc[:, list(SCF_AUTO_LOAN_AMOUNT_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    rates = joined.loc[:, list(SCF_AUTO_LOAN_RATE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    )
    amounts = amounts.fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    rates = rates.fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64) / 10_000.0

    donor = predictors.reset_index(drop=True)
    donor["auto_loan_balance"] = amounts.sum(axis=1)
    donor["auto_loan_interest"] = np.sum(amounts * rates, axis=1)
    donor = donor.loc[
        donor[_DONOR_WEIGHT_COLUMN] > 0,
        [
            *SCF_WEALTH_PREDICTORS,
            "auto_loan_balance",
            "auto_loan_interest",
            _DONOR_WEIGHT_COLUMN,
        ],
    ]
    return donor.reset_index(drop=True)


def _scf_reference_person_mask(person: pd.DataFrame) -> np.ndarray:
    """Select one recipient per household using the retired SCF rules."""

    required = {
        _HOUSEHOLD_ID_COLUMN,
        "age",
        "is_female",
        "A_MARITL",
        "A_LINENO",
    }
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(
            f"SCF auto-loan reference-person selection missing column(s): {missing}."
        )

    working = pd.DataFrame(
        {
            "household_id": person[_HOUSEHOLD_ID_COLUMN].to_numpy(),
            "age": pd.to_numeric(person["age"], errors="coerce").fillna(0.0).to_numpy(),
            "is_female": person["is_female"].astype(bool).to_numpy(),
            "is_married": pd.to_numeric(person["A_MARITL"], errors="coerce")
            .isin([1, 2])
            .to_numpy(),
            "position": np.arange(len(person)),
        }
    )
    mask = np.zeros(len(person), dtype=bool)
    for _, group in working.groupby("household_id", sort=False):
        adults = group.loc[group["age"] >= 18]
        if adults.empty:
            chosen = int(group.loc[group["age"].idxmax(), "position"])
        elif len(adults) == 1:
            chosen = int(adults.iloc[0]["position"])
        elif len(adults) == 2 and bool(adults["is_married"].all()):
            if adults["is_female"].nunique() == 1:
                chosen = int(adults.loc[adults["age"].idxmax(), "position"])
            else:
                chosen = int(adults.loc[~adults["is_female"]].iloc[0]["position"])
        else:
            chosen = int(adults.loc[adults["age"].idxmax(), "position"])
        mask[chosen] = True
    return mask


def _recipient_household_predictor_table(frame: Frame) -> pd.DataFrame:
    person = frame.table("person")
    household = frame.table("household")
    missing = sorted(set(_REQUIRED_RECIPIENT_PERSON_COLUMNS) - set(person.columns))
    if missing:
        raise ValueError(
            f"US SCF auto-loan imputation requires recipient person column(s): "
            f"{missing}."
        )

    mask = _scf_reference_person_mask(person)
    per_person = pd.DataFrame(index=person.index)
    per_person["household_id"] = person[_HOUSEHOLD_ID_COLUMN].to_numpy()
    per_person["age"] = pd.to_numeric(person["age"], errors="coerce").fillna(0.0)
    per_person["is_female"] = person["is_female"].astype(bool).astype(float)
    per_person["cps_race"] = _recipient_cps_race(person)
    per_person["is_married"] = (
        pd.to_numeric(person["A_MARITL"], errors="coerce").isin([1, 2]).astype(float)
    )
    per_person["own_children_in_household"] = _own_children_in_household(person)

    income = pd.DataFrame(
        {
            "household_id": person[_HOUSEHOLD_ID_COLUMN].to_numpy(),
            "employment_income": np.maximum(
                pd.to_numeric(person["employment_income_before_lsr"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64),
                0.0,
            ),
            "interest_dividend_income": _sum_present(
                person, _RECIPIENT_INTEREST_DIVIDEND_COLUMNS
            ),
            "social_security_pension_income": _sum_present(
                person, _RECIPIENT_SS_PENSION_COLUMNS
            ),
        }
    )
    household_income = income.groupby("household_id", sort=False).sum()
    reference = per_person.loc[mask].set_index("household_id")
    if reference.index.duplicated().any():
        raise ValueError("SCF auto-loan receiver selected multiple reference persons.")
    reference = reference.join(household_income, how="left")

    household_ids = household["household_id"].to_numpy()
    missing_ids = sorted(set(household_ids) - set(reference.index))
    extra_ids = sorted(set(reference.index) - set(household_ids))
    if missing_ids or extra_ids:
        raise ValueError(
            "SCF auto-loan receiver household alignment failed: "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}."
        )
    return reference.reindex(household_ids).loc[:, list(SCF_WEALTH_PREDICTORS)]


def impute_us_scf_auto_loans(
    frame: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Draw legacy auto balance/interest jointly, once per household."""

    from microcosm.fit import QRF

    targets = ("auto_loan_balance", "auto_loan_interest")
    required = (*SCF_WEALTH_PREDICTORS, *targets, _DONOR_WEIGHT_COLUMN)
    missing = [column for column in required if column not in donor.columns]
    if missing:
        raise ValueError(f"SCF auto-loan donor table missing column(s): {missing}.")
    fit_frame = donor.loc[:, list(required)].copy()
    for column in required:
        fit_frame[column] = pd.to_numeric(fit_frame[column], errors="coerce").fillna(
            0.0
        )
    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        fit_frame,
        predictors=list(SCF_WEALTH_PREDICTORS),
        targets=list(targets),
        weights=_DONOR_WEIGHT_COLUMN,
    )
    drawn = fitted.predict(_recipient_household_predictor_table(frame))
    result = pd.DataFrame(index=frame.table("household").index)
    for column in targets:
        result[column] = np.maximum(np.asarray(drawn[column], dtype=np.float64), 0.0)
    return result


def qualified_auto_loan_interest_proxy(
    auto_loan_interest: np.ndarray,
    household_weights: np.ndarray,
    *,
    annual_issuance_target: float = QUALIFIED_AUTO_LOAN_ANNUAL_ISSUANCE_TARGET,
) -> tuple[np.ndarray, float]:
    """Return expected qualifying interest and the applied incidence share."""

    interest = np.maximum(np.asarray(auto_loan_interest, dtype=np.float64), 0.0)
    weights = np.asarray(household_weights, dtype=np.float64)
    if interest.shape != weights.shape:
        raise ValueError("Auto-loan interest and household weights must align.")
    positive_mass = float(weights[interest > 0].sum())
    share = (
        min(float(annual_issuance_target) / positive_mass, 1.0)
        if positive_mass > 0
        else 0.0
    )
    return interest * share, share


def _surface_has_signal(household: pd.DataFrame) -> bool:
    if not all(column in household for column in US_SCF_AUTO_LOAN_OUTPUT_COLUMNS):
        return False
    for column in US_SCF_AUTO_LOAN_OUTPUT_COLUMNS:
        values = pd.to_numeric(household[column], errors="coerce").dropna()
        if values.nunique() < 2 or not (values > 0).any():
            return False
    qualified = pd.to_numeric(
        household["qualified_passenger_vehicle_loan_interest"], errors="coerce"
    ).fillna(0.0)
    interest = pd.to_numeric(household["auto_loan_interest"], errors="coerce").fillna(
        0.0
    )
    return bool((qualified <= interest).all())


def with_us_scf_auto_loan_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    scf_auto_loan_donor: pd.DataFrame,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> Frame:
    """Restore both legacy and OBBBA auto-loan inputs on ``household``."""

    if frame.schema != US_SCHEMA:
        raise ValueError("US SCF auto-loan inputs require the US schema.")
    us_scf_auto_loans_stage_spec()
    household = frame.table("household")
    if _surface_has_signal(household):
        return frame

    imputed = impute_us_scf_auto_loans(
        frame,
        scf_auto_loan_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    qualified, _ = qualified_auto_loan_interest_proxy(
        imputed["auto_loan_interest"].to_numpy(dtype=np.float64),
        frame.weights_for("household").values,
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"]["auto_loan_balance"] = imputed["auto_loan_balance"].to_numpy(
        dtype=np.float64
    )
    tables["household"]["auto_loan_interest"] = imputed["auto_loan_interest"].to_numpy(
        dtype=np.float64
    )
    tables["household"]["qualified_passenger_vehicle_loan_interest"] = qualified
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def us_scf_auto_loans_summary(frame: Frame) -> dict[str, object]:
    """Weighted incidence/totals and relation diagnostics for the family."""

    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    total_weight = float(weights.sum())
    values = {
        column: pd.to_numeric(household[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        for column in US_SCF_AUTO_LOAN_OUTPUT_COLUMNS
        if column in household
    }
    interest = values.get("auto_loan_interest", np.zeros(len(household)))
    qualified = values.get(
        "qualified_passenger_vehicle_loan_interest", np.zeros(len(household))
    )
    positive_mass = float(weights[interest > 0].sum())
    qualified_total = float(np.dot(qualified, weights))
    interest_total = float(np.dot(interest, weights))
    return {
        "auto_loan_balance_weighted_total": float(
            np.dot(values.get("auto_loan_balance", np.zeros(len(household))), weights)
        ),
        "auto_loan_interest_weighted_total": interest_total,
        "qualified_interest_weighted_total": qualified_total,
        "auto_loan_interest_nonzero_share": (
            positive_mass / total_weight if total_weight > 0 else 0.0
        ),
        "qualified_interest_share": (
            qualified_total / interest_total if interest_total > 0 else 0.0
        ),
        "qualified_exceeds_interest_count": int((qualified > interest).sum()),
        "negative_counts": {
            column: int((column_values < 0).sum())
            for column, column_values in values.items()
        },
        "unique_counts": {
            column: int(pd.Series(column_values).nunique())
            for column, column_values in values.items()
        },
        "auto_loan_nonzero_share_band": list(_AUTO_LOAN_NONZERO_SHARE_BAND),
    }


def us_scf_auto_loans_signal_gate(frame: Frame) -> GateResult:
    """Require nonnegative, nonconstant legacy and OBBBA auto-loan signal."""

    household = frame.table("household")
    missing = [
        column
        for column in US_SCF_AUTO_LOAN_OUTPUT_COLUMNS
        if column not in household.columns
    ]
    if missing:
        return GateResult(
            name="scf_auto_loans_signal",
            passed=False,
            failures=(f"household columns missing: {missing}.",),
            details={"missing": missing},
        )

    summary = us_scf_auto_loans_summary(frame)
    failures: list[str] = []
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(f"{column}: constant column carries no signal.")
    for column, count in summary["negative_counts"].items():
        if count:
            failures.append(f"{column}: {count} negative value(s).")
    if summary["qualified_exceeds_interest_count"]:
        failures.append(
            "qualified_passenger_vehicle_loan_interest exceeds auto_loan_interest "
            f"on {summary['qualified_exceeds_interest_count']} household(s)."
        )
    share = float(summary["auto_loan_interest_nonzero_share"])
    low, high = _AUTO_LOAN_NONZERO_SHARE_BAND
    if not (low <= share <= high):
        failures.append(
            f"auto-loan nonzero share {share:.3f} outside plausibility band "
            f"[{low}, {high}]."
        )
    if float(summary["qualified_interest_weighted_total"]) <= 0:
        failures.append("qualified auto-loan interest weighted total is zero.")
    return GateResult(
        name="scf_auto_loans_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
