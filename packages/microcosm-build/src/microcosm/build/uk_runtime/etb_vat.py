"""UK ETB VAT imputation stage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.donor_uprating import (
    apply_donor_uprating,
    donor_uprating_factors,
    uprating_operation,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.household_income import (
    UK_HBAI_GROSS_INCOME_VARIABLES,
    etb_disposable_income,
    household_level,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import extreme_count
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    UKSupportClipResult,
    support_clip_to_donor_with_receipt,
)
from microcosm.build.uk_runtime.uc_relationships import household_family_role_counts
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

ETB_FILENAME = "householdv2_1977-2024.tab"
ETB_SHA256 = "d0e94ebc92e85ca1b9fb3a7353dcaf41db2c5110c9f07c7793dc8c0b695250d8"
ETB_SIZE_BYTES = 216_967_663
#: The income the rate is drawn on: the donor's ETB disposable income
#: (``disinc``, before indirect taxes) and the recipient's on the same concept,
#: its gross cash income less income tax, National Insurance and council tax
#: (``household_income.etb_disposable_income``; microcosm#1164). The engine's
#: ``household_net_income`` it replaces also takes off the indirect taxes and
#: keeps the in-kind NHS and education spending, which read recipients about a
#: quarter richer than the donor at the median.
UK_ETB_VAT_INCOME_PREDICTOR = "etb_disposable_income"
UK_ETB_VAT_PREDICTORS = (
    "is_adult",
    "is_child",
    "is_SP_age",
    UK_ETB_VAT_INCOME_PREDICTOR,
)
#: The ETB donor's ``adults`` and ``childs`` count a household's adults and
#: dependent children, so the recipient counts are its FRS claimants and
#: partners and its other members (``is_uc_claimant``), not the engine's
#: age-18 split (uk-data#486, microcosm#1095). Only the rest materialize.
UK_ETB_FAMILY_ROLE_COUNTS = ("is_adult", "is_child")
UK_ETB_VAT_ENGINE_PREDICTORS = ("is_SP_age", *UK_HBAI_GROSS_INCOME_VARIABLES)
UK_ETB_VAT_OUTPUT_COLUMNS = ("full_rate_vat_expenditure_rate",)
#: The full-rate VAT expenditure rate is a share of expenditure after VAT, so
#: it lies in [0, 1] (microcosm#1225, uk-data#572). The raw ETB accounts carry households whose
#: VAT equals or exceeds their disposable expenditure, which leave nothing for
#: the rate to be a share of; they are dropped from the donor, and the rates
#: above one, where little expenditure is left after VAT, are capped at one.
UK_ETB_VAT_RATE_BOUNDS = (0.0, 1.0)
UK_ETB_VAT_NONNEGATIVE_OUTPUT_COLUMNS = UK_ETB_VAT_OUTPUT_COLUMNS
UK_ETB_VAT_FIT_NAME = "uk_etb_2023_vat:full_rate_vat_expenditure_rate"
UK_ETB_VAT_STAGE_NAME = "etb_vat"


@dataclass
class UKETBVATResult:
    """Transformed frame, donor-support clip and donor rate receipts."""

    frame: Frame
    support_clip: UKSupportClipReceipt
    donor_rate: dict[str, object] | None = None
    donor_uprating: dict[str, object] | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
            "stage": UK_ETB_VAT_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
        }
        if self.donor_rate is not None:
            evidence["donor_rate"] = dict(self.donor_rate)
        if self.donor_uprating is not None:
            evidence["donor_uprating"] = dict(self.donor_uprating)
        return evidence


#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_ETB_VAT_MASS_CONSERVATION_REASON = "ETB VAT expenditure-rate imputation on the source spine: household weights pass through unchanged and total household mass is conserved."


@dataclass
class UKETBVATStageTransform:
    stage: SourceStageSpec
    engine: object
    etb_tab_path: str | Path | None = None
    donor: pd.DataFrame | None = None
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_result: UKETBVATResult | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        return (
            () if self.last_fit_weight_records is None else self.last_fit_weight_records
        )

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        config = etb_vat_configuration(self.stage)
        raw = (
            self.donor
            if self.donor is not None
            else read_pinned_tab(
                _require_path(self.etb_tab_path), self.stage.artifacts[0]
            )
        )
        donor = clean_etb_vat_table(raw, **config)
        donor_rate = donor.attrs.get(UK_ETB_VAT_DONOR_RATE_RECEIPT)
        uprating = uprating_operation(self.stage)
        uprating_receipt = None
        if uprating is not None:
            factors, uprating_receipt = donor_uprating_factors(uprating)
            donor = apply_donor_uprating(donor, factors)
        predictors = recipient_predictors(frame, self.engine)
        imputed, record = impute_etb_vat(
            donor,
            predictors,
            seed=_qrf_seed(self.stage),
            household_ids=frame.table("household")["household_id"].to_numpy(),
            salt=_qrf_salt(self.stage),
        )
        clip_result = support_clip_to_donor(imputed, donor)
        imputed = clip_result.clipped
        household = frame.table("household").copy()
        household["full_rate_vat_expenditure_rate"] = imputed[
            "full_rate_vat_expenditure_rate"
        ].to_numpy()
        result = uk_national_frame(
            person=frame.table("person").copy(),
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_ETB_VAT_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = (record,)
        self.last_result = UKETBVATResult(
            frame=result,
            support_clip=clip_result.receipt,
            donor_rate=donor_rate,
            donor_uprating=uprating_receipt,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_ETB_VAT_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


#: ``donor.attrs`` key of the receipt ``clean_etb_vat_table`` writes: the
#: declared rate bounds, the rows dropped for leaving no expenditure after VAT
#: and the rows capped at the upper bound.
UK_ETB_VAT_DONOR_RATE_RECEIPT = "donor_rate"


def clean_etb_vat_table(
    raw: pd.DataFrame,
    *,
    year: int | None = None,
    standard_rate: float | None = None,
    reduced_rate_share: float | None = None,
    rate_bounds: tuple[float, float] = UK_ETB_VAT_RATE_BOUNDS,
) -> pd.DataFrame:
    resource = _load_etb_vat_resource()
    if year is None:
        year = int(resource["vat"]["standard_rate"]["period"])
    if standard_rate is None:
        standard_rate = float(resource["vat"]["standard_rate"]["value"])
    if reduced_rate_share is None:
        reduced_rate_share = float(resource["vat"]["reduced_rate_share"]["value"])
    if not np.isfinite(standard_rate) or standard_rate <= 0:
        raise ValueError("VAT standard_rate must be positive and finite.")
    if not np.isfinite(reduced_rate_share):
        raise ValueError("VAT reduced_rate_share must be finite.")
    lower, upper = _checked_rate_bounds(rate_bounds)
    data = raw.replace(r"^\s*$", np.nan, regex=True)
    required = [
        "year",
        "adults",
        "childs",
        "noretd",
        "disinc",
        "totvat",
        "expdis",
        "hhold_adj_weight",
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"ETB VAT donor is missing required column(s): {missing}.")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data[data["year"] == year].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required)
    # A household whose VAT equals or exceeds its disposable expenditure leaves
    # no expenditure for the rate to be a share of (uk-data#572): no donor.
    denominator = data["expdis"] - data["totvat"]
    no_expenditure_after_vat = denominator <= 0
    data = data[~no_expenditure_after_vat]
    denominator = denominator[~no_expenditure_after_vat]
    train = pd.DataFrame()
    train["is_adult"] = data["adults"]
    train["is_child"] = data["childs"]
    train["is_SP_age"] = data["noretd"]
    train[UK_ETB_VAT_INCOME_PREDICTOR] = data["disinc"] * 52
    train["weight"] = data["hhold_adj_weight"]
    rate = (data["totvat"] * (1 - reduced_rate_share) / standard_rate) / denominator
    if not np.isfinite(rate).all():
        raise ValueError("ETB VAT donor produced a non-finite VAT expenditure rate.")
    if (rate < lower).any():
        raise ValueError(
            "ETB VAT donor produced a VAT expenditure rate below the declared "
            f"lower bound {lower}: totvat is negative."
        )
    above = rate > upper
    train["full_rate_vat_expenditure_rate"] = rate.clip(upper=upper)
    train = train.dropna()
    train.attrs[UK_ETB_VAT_DONOR_RATE_RECEIPT] = {
        "rate_bounds": [lower, upper],
        "rows_dropped_no_expenditure_after_vat": extreme_count(
            int(no_expenditure_after_vat.sum())
        ),
        "rows_capped_at_upper_bound": extreme_count(int(above.sum())),
    }
    return train


def _checked_rate_bounds(bounds: object) -> tuple[float, float]:
    """The declared rate bounds, which must be the share-of-expenditure [0, 1]."""

    try:
        lower, upper = (float(value) for value in bounds)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"ETB VAT rate_bounds must be a [lower, upper] pair, got {bounds!r}."
        ) from error
    if (lower, upper) != UK_ETB_VAT_RATE_BOUNDS:
        raise ValueError(
            f"ETB VAT rate_bounds must be {list(UK_ETB_VAT_RATE_BOUNDS)}: the rate "
            f"is a share of expenditure after VAT; got {[lower, upper]}."
        )
    return lower, upper


def _load_etb_vat_resource() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("etb_policy_anchors.json")
        .read_text(encoding="utf-8")
    )


def etb_vat_configuration(
    stage: SourceStageSpec | None = None,
) -> dict[str, object]:
    """Load VAT settings from the committed resource and validate the manifest."""

    resource = _load_etb_vat_resource()["vat"]
    config: dict[str, object] = {
        "year": int(resource["standard_rate"]["period"]),
        "standard_rate": float(resource["standard_rate"]["value"]),
        "reduced_rate_share": float(resource["reduced_rate_share"]["value"]),
    }
    if stage is not None:
        derive = next(
            operation for operation in stage.operations if operation.kind == "derive"
        )
        for key, value in config.items():
            declared = derive.parameters.get(key)
            if declared is not None and float(declared) != float(value):
                raise ValueError(
                    f"ETB VAT manifest {key}={declared!r} disagrees with "
                    f"the committed anchor value {value!r}."
                )
        bounds = derive.parameters.get("rate_bounds")
        if bounds is None:
            raise ValueError(
                f"{stage.stage}: the derive operation must declare rate_bounds, "
                "the range the VAT expenditure rate is a share in."
            )
        config["rate_bounds"] = _checked_rate_bounds(bounds)
    return config


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize ETB VAT recipient predictors at household grain.

    Engine variables materialize at their native entity (is_SP_age is a
    person boolean) and aggregate to household by person_household_id — direct
    engine arrays would crash the licensed build on the person/household
    length mismatch (the E5 review class). The adult and child counts are the
    household's FRS family roles, and the income is its disposable income on
    the ETB's concept (``etb_disposable_income``).
    """

    materialized = engine.materialize(
        frame, UK_ETB_VAT_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    result = pd.DataFrame(index=household.index)
    adults, children = household_family_role_counts(person, household)
    result["is_adult"] = adults
    result["is_child"] = children
    result["is_SP_age"] = household_level(
        frame, engine, "is_SP_age", materialized["is_SP_age"]
    )
    result[UK_ETB_VAT_INCOME_PREDICTOR] = etb_disposable_income(
        frame, engine, materialized
    )
    return result.loc[:, list(UK_ETB_VAT_PREDICTORS)]


def impute_etb_vat(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    seed: int,
    household_ids: np.ndarray,
    salt: str,
    n_estimators: int = 100,
) -> tuple[pd.DataFrame, FitWeightRecord]:
    """Draw the rate at uniforms keyed on each household's id.

    Each household's quantile and sign uniforms come from
    ``stable_identity_uniforms(household_ids, seed, salt)``, so a household
    keeps its draw however the frame is ordered (microcosm#1164; the model's
    own generator drew in row order).
    """

    from microcosm.fit import RegimeGatedQRF

    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    fitted = model.fit(
        donor,
        list(UK_ETB_VAT_PREDICTORS),
        list(UK_ETB_VAT_OUTPUT_COLUMNS),
        weights="weight",
    )
    ids = np.asarray(household_ids)
    if len(ids) != len(recipient):
        raise ValueError("household_ids must align with the recipient predictors.")
    drawn = fitted.predict_from_uniforms(
        recipient.loc[:, list(UK_ETB_VAT_PREDICTORS)].reset_index(drop=True),
        quantiles={
            target: stable_identity_uniforms(ids, seed=seed, salt=f"{salt}:{target}")
            for target in UK_ETB_VAT_OUTPUT_COLUMNS
        },
        sign_uniforms={
            target: stable_identity_uniforms(
                ids, seed=seed, salt=f"{salt}:{target}:sign"
            )
            for target in UK_ETB_VAT_OUTPUT_COLUMNS
        },
    )
    drawn.index = recipient.index
    return drawn, FitWeightRecord(UK_ETB_VAT_FIT_NAME, "explicit")


def support_clip_to_donor(
    draws: pd.DataFrame, donor: pd.DataFrame
) -> UKSupportClipResult:
    return support_clip_to_donor_with_receipt(
        draws,
        donor,
        columns=UK_ETB_VAT_OUTPUT_COLUMNS,
        stage=UK_ETB_VAT_STAGE_NAME,
    )


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    values = donor["full_rate_vat_expenditure_rate"]
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {}
    return {
        "full_rate_vat_expenditure_rate": (float(finite.min()), float(finite.max()))
    }


def _qrf_seed(stage: SourceStageSpec) -> int:
    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf":
            return int(operation.parameters.get("seed", 0))
    return 0


def _qrf_salt(stage: SourceStageSpec) -> str:
    """The declared salt of the identity-keyed draw uniforms."""

    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf":
            salt = operation.parameters.get("salt")
            if isinstance(salt, str) and salt:
                return salt
    raise ValueError(
        f"{UK_ETB_VAT_STAGE_NAME} must declare a fit_weighted_qrf salt for its "
        "identity-keyed draws."
    )


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("ETB VAT stage requires a caller-supplied ETB tab path.")
    return Path(path).expanduser().resolve()
