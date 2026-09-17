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
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    UKSupportClipResult,
    support_clip_to_donor_with_receipt,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

ETB_FILENAME = "householdv2_1977-2024.tab"
ETB_SHA256 = "d0e94ebc92e85ca1b9fb3a7353dcaf41db2c5110c9f07c7793dc8c0b695250d8"
ETB_SIZE_BYTES = 216_967_663
UK_ETB_VAT_PREDICTORS = (
    "is_adult",
    "is_child",
    "is_SP_age",
    "household_net_income",
)
UK_ETB_VAT_OUTPUT_COLUMNS = ("full_rate_vat_expenditure_rate",)
# The donor-realized support includes negative rates (totvat can exceed expdis
# in the raw ETB accounts), so the output stays out of the nonnegative gate and
# the support gate is the guard — the net_financial_wealth precedent.
UK_ETB_VAT_NONNEGATIVE_OUTPUT_COLUMNS: tuple[str, ...] = ()
UK_ETB_VAT_FIT_NAME = "uk_etb_2023_vat:full_rate_vat_expenditure_rate"
UK_ETB_VAT_STAGE_NAME = "etb_vat"


@dataclass
class UKETBVATResult:
    """Transformed frame and donor-support clip receipt."""

    frame: Frame
    support_clip: UKSupportClipReceipt

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_ETB_VAT_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
        }


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
        predictors = recipient_predictors(frame, self.engine)
        imputed, record = impute_etb_vat(donor, predictors, seed=_qrf_seed(self.stage))
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
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = (record,)
        self.last_result = UKETBVATResult(
            frame=result,
            support_clip=clip_result.receipt,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_ETB_VAT_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def clean_etb_vat_table(
    raw: pd.DataFrame,
    *,
    year: int | None = None,
    standard_rate: float | None = None,
    reduced_rate_share: float | None = None,
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
    train = pd.DataFrame()
    train["is_adult"] = data["adults"]
    train["is_child"] = data["childs"]
    train["is_SP_age"] = data["noretd"]
    train["household_net_income"] = data["disinc"] * 52
    train["weight"] = data["hhold_adj_weight"]
    denominator = data["expdis"] - data["totvat"]
    if (denominator == 0).any():
        raise ValueError("ETB VAT donor contains zero disposable-expenditure denominators.")
    train["full_rate_vat_expenditure_rate"] = (
        data["totvat"] * (1 - reduced_rate_share) / standard_rate
    ) / denominator
    if not np.isfinite(train["full_rate_vat_expenditure_rate"]).all():
        raise ValueError("ETB VAT donor produced a non-finite VAT expenditure rate.")
    return train.dropna()


def _load_etb_vat_resource() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("etb_policy_anchors.json")
        .read_text(encoding="utf-8")
    )


def etb_vat_configuration(
    stage: SourceStageSpec | None = None,
) -> dict[str, float | int]:
    """Load VAT settings from the committed resource and validate the manifest."""

    resource = _load_etb_vat_resource()["vat"]
    config: dict[str, float | int] = {
        "year": int(resource["standard_rate"]["period"]),
        "standard_rate": float(resource["standard_rate"]["value"]),
        "reduced_rate_share": float(resource["reduced_rate_share"]["value"]),
    }
    if stage is not None:
        derive = next(
            operation
            for operation in stage.operations
            if operation.kind == "derive"
        )
        for key, value in config.items():
            declared = derive.parameters.get(key)
            if declared is not None and float(declared) != float(value):
                raise ValueError(
                    f"ETB VAT manifest {key}={declared!r} disagrees with "
                    f"the committed anchor value {value!r}."
                )
    return config


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize ETB VAT recipient predictors at household grain.

    Predictors materialize at their native entity (is_adult / is_child /
    is_SP_age are person booleans) and aggregate to household by
    person_household_id — direct engine arrays would crash the licensed
    build on the person/household length mismatch (the E5 review class).
    """

    materialized = engine.materialize(
        frame, UK_ETB_VAT_PREDICTORS, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    result = pd.DataFrame(index=household.index)
    for predictor in UK_ETB_VAT_PREDICTORS:
        declared = str(engine.variable_metadata(predictor).entity)
        values = np.asarray(materialized[predictor])
        if declared == "household":
            result[predictor] = values
        elif declared == "person":
            summed = (
                pd.Series(values.astype(float))
                .groupby(person["person_household_id"].to_numpy())
                .sum()
            )
            result[predictor] = (
                summed.reindex(household["household_id"]).fillna(0.0).to_numpy()
            )
        else:
            raise ValueError(f"unsupported ETB VAT predictor entity {declared!r}.")
    return result


def impute_etb_vat(
    donor: pd.DataFrame, recipient: pd.DataFrame, *, seed: int, n_estimators: int = 100
) -> tuple[pd.DataFrame, FitWeightRecord]:
    from microcosm.fit import RegimeGatedQRF

    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    fitted = model.fit(
        donor,
        list(UK_ETB_VAT_PREDICTORS),
        list(UK_ETB_VAT_OUTPUT_COLUMNS),
        weights="weight",
    )
    return fitted.predict(recipient), FitWeightRecord(UK_ETB_VAT_FIT_NAME, "explicit")


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


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("ETB VAT stage requires a caller-supplied ETB tab path.")
    return Path(path).expanduser().resolve()
