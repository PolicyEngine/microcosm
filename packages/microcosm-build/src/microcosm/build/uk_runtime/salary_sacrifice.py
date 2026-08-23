"""UK salary-sacrifice QRF and headcount support conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_structure import (
    HOUSEHOLD_IS_CGT_BAND_DONOR,
    HOUSEHOLD_IS_CGT_CLONE,
    _assert_closed_world_operations,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import support_channel_column
from microcosm.frame import Frame, MassChangeRecord

QRF: Any | None = None

SALSAC_PREDICTORS = ("age", "employment_income")
SALSAC_OUTPUT = "pension_contributions_via_salary_sacrifice"
SALSAC_STAGE_TARGET = 5_400_000.0
SALSAC_HMRC_ANCHOR = 7_700_000.0
SALSAC_ABOVE_2000_ANCHOR = 3_300_000.0
SALSAC_BELOW_2000_ANCHOR = 4_300_000.0
SALSAC_STAGING_RATIO = SALSAC_STAGE_TARGET / SALSAC_HMRC_ANCHOR
SALSAC_RATE_CAP = 0.5
SALSAC_QRF_SEED = 42
SALSAC_QRF_ESTIMATORS = 100
SALSAC_CONVERSION_SEED = 2024
SALSAC_CONVERSION_SALT = "salary_sacrifice_conversion"
SALSAC_MASS_CHANGE_REASON = (
    "Salary-sacrifice support stage rewrites pension columns only; household "
    "rows and typed household weights pass through and total household mass "
    "is conserved."
)


def load_salary_sacrifice_anchor() -> Mapping[str, Any]:
    """Load the committed HMRC salary-sacrifice anchor."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("salary_sacrifice_anchor.json")
        .read_text(encoding="utf-8")
    )


@dataclass(frozen=True)
class UKSalarySacrificeResult:
    """Transformed frame and the full headcount executed-effect receipt."""

    frame: Frame
    training_rows: int
    prediction_rows: int
    pre_headcount: float
    post_headcount: float
    shortfall: float
    donor_pool_mass: float
    rate: float
    cap_bound: bool
    converted_rows: int
    converted_mass: float
    moved_amount: float
    expected_converted_mass: float
    realization_deviation: float

    def evidence(self) -> dict[str, object]:
        return {
            "stage": "salary_sacrifice",
            "qrf": {
                "training_rows": self.training_rows,
                "prediction_rows": self.prediction_rows,
                "seed": SALSAC_QRF_SEED,
            },
            "headcount_receipt": {
                "target": SALSAC_STAGE_TARGET,
                "pre_headcount": self.pre_headcount,
                "post_headcount": self.post_headcount,
                "shortfall": self.shortfall,
                "donor_pool_mass": self.donor_pool_mass,
                "rate": self.rate,
                "rate_cap": SALSAC_RATE_CAP,
                "cap_bound": self.cap_bound,
                "converted_rows": self.converted_rows,
                "converted_mass": self.converted_mass,
                "moved_amount": self.moved_amount,
                "expected_converted_mass": self.expected_converted_mass,
                "realization_deviation": self.realization_deviation,
            },
        }


@dataclass(frozen=True)
class UKSalarySacrificeStageTransform:
    """Whole-stage transform for salary-sacrifice support."""

    stage: SourceStageSpec
    anchor: Mapping[str, Any] | None = None
    last_result: UKSalarySacrificeResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        resource = self.anchor or load_salary_sacrifice_anchor()
        _assert_salary_sacrifice_stage_parameters(self.stage, anchor=resource)
        result = impute_salary_sacrifice(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (SALSAC_OUTPUT, "employee_pension_contributions")

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def impute_salary_sacrifice(frame: Frame) -> UKSalarySacrificeResult:
    """Fit on the asked base-FRS subset, then create additional SS support."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    required_person = {
        "person_id",
        "person_household_id",
        "age",
        "employment_income",
        "salary_sacrifice_asked",
        SALSAC_OUTPUT,
        "employee_pension_contributions",
    }
    missing = sorted(required_person - set(person.columns))
    if missing:
        raise ValueError(f"Salary-sacrifice person columns missing: {missing}.")
    households = household.set_index("household_id")
    person_households = person["person_household_id"]
    if not person_households.isin(households.index).all():
        raise ValueError("Salary-sacrifice people must map to a household.")
    channel_column = support_channel_column("household")
    if channel_column not in household.columns:
        raise ValueError(
            f"Salary-sacrifice training requires household {channel_column!r}."
        )
    channels = person_households.map(households[channel_column])
    clones = person_households.map(
        households.get(HOUSEHOLD_IS_CGT_CLONE, pd.Series(False, index=households.index))
    ).fillna(False)
    donors = person_households.map(
        households.get(
            HOUSEHOLD_IS_CGT_BAND_DONOR,
            pd.Series(False, index=households.index),
        )
    ).fillna(False)
    asked = pd.to_numeric(person["salary_sacrifice_asked"], errors="coerce")
    if asked.isna().any():
        raise ValueError("salary_sacrifice_asked contains non-numeric values.")
    training_mask = (
        channels.eq("frs") & ~clones.astype(bool) & ~donors.astype(bool) & asked.eq(1)
    )
    if not training_mask.any():
        raise ValueError("Salary-sacrifice QRF has no eligible asked FRS rows.")
    predict_mask = ~asked.eq(1)
    numeric = person.loc[:, [*SALSAC_PREDICTORS, SALSAC_OUTPUT]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Salary-sacrifice QRF columns must be finite numeric values.")
    household_weights = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"],
    )
    person_weights = person_households.map(household_weights).to_numpy(dtype=float)
    training = numeric.loc[training_mask, [*SALSAC_PREDICTORS, SALSAC_OUTPUT]].copy()
    training["_fit_weight"] = person_weights[training_mask.to_numpy()]
    model = _qrf_class()(n_estimators=SALSAC_QRF_ESTIMATORS, seed=SALSAC_QRF_SEED)
    fitted = model.fit(
        training,
        list(SALSAC_PREDICTORS),
        [SALSAC_OUTPUT],
        weights="_fit_weight",
    )
    if predict_mask.any():
        predictions = fitted.predict(numeric.loc[predict_mask, list(SALSAC_PREDICTORS)])
        predicted = pd.to_numeric(predictions[SALSAC_OUTPUT], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(predicted).all():
            raise ValueError("Salary-sacrifice QRF produced non-finite predictions.")
        person.loc[predict_mask, SALSAC_OUTPUT] = np.maximum(0.0, predicted)
    final_ss = pd.to_numeric(person[SALSAC_OUTPUT], errors="coerce").to_numpy(
        dtype=float, copy=True
    )
    employee = pd.to_numeric(
        person["employee_pension_contributions"], errors="coerce"
    ).to_numpy(dtype=float, copy=True)
    employment_income = pd.to_numeric(
        person["employment_income"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(final_ss).all() or (final_ss < 0.0).any():
        raise ValueError("Salary-sacrifice amounts must be finite and non-negative.")
    if not np.isfinite(employee).all() or (employee < 0.0).any():
        raise ValueError("Employee-pension amounts must be finite and non-negative.")
    has_ss = final_ss > 0.0
    pre_headcount = float(person_weights[has_ss].sum())
    shortfall = max(0.0, SALSAC_STAGE_TARGET - pre_headcount)
    donor_pool = (employee > 0.0) & ~has_ss & (employment_income > 0.0)
    donor_pool_mass = float(person_weights[donor_pool].sum())
    uncapped_rate = shortfall / donor_pool_mass if donor_pool_mass > 0.0 else 0.0
    rate = min(SALSAC_RATE_CAP, uncapped_rate)
    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(),
        seed=SALSAC_CONVERSION_SEED,
        salt=SALSAC_CONVERSION_SALT,
    )
    converted = donor_pool & (draws < rate)
    moved_amount = float(employee[converted].sum())
    final_ss[converted] = employee[converted]
    employee[converted] = 0.0
    person[SALSAC_OUTPUT] = final_ss
    person["employee_pension_contributions"] = employee
    post_headcount = float(person_weights[final_ss > 0.0].sum())
    converted_mass = float(person_weights[converted].sum())
    total = frame.weights_for("household").total
    mass_receipt = MassChangeRecord(
        entity="household",
        old_total=total,
        new_total=total,
        declared_factor=1.0,
        reason=SALSAC_MASS_CHANGE_REASON,
    )
    result_frame = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=(*frame.mass_log, mass_receipt),
    )
    validate_uk_national_frame(result_frame)
    return UKSalarySacrificeResult(
        frame=result_frame,
        training_rows=int(training_mask.sum()),
        prediction_rows=int(predict_mask.sum()),
        pre_headcount=pre_headcount,
        post_headcount=post_headcount,
        shortfall=shortfall,
        donor_pool_mass=donor_pool_mass,
        rate=rate,
        cap_bound=uncapped_rate > SALSAC_RATE_CAP,
        converted_rows=int(converted.sum()),
        converted_mass=converted_mass,
        moved_amount=moved_amount,
        expected_converted_mass=rate * donor_pool_mass,
        realization_deviation=(
            (converted_mass - rate * donor_pool_mass) / (rate * donor_pool_mass)
            if rate * donor_pool_mass > 0.0
            else 0.0
        ),
    )


def _qrf_class():
    if QRF is not None:
        return QRF
    return import_module("microcosm.fit").QRF


def _assert_salary_sacrifice_stage_parameters(
    stage: SourceStageSpec,
    *,
    anchor: Mapping[str, Any],
) -> None:
    """Bind all stage parameters closed-world; result evidence supplies arm 2."""

    _assert_closed_world_operations(
        stage,
        (
            (
                "fit_weighted_qrf",
                {
                    "training_population": (
                        "support_channel == frs and not capital-gains clone and "
                        "not CGT band donor and salary_sacrifice_asked == 1"
                    ),
                    "target_population": "salary_sacrifice_asked != 1 frame-wide",
                    "predictors": list(SALSAC_PREDICTORS),
                    "targets": [SALSAC_OUTPUT],
                    "weights": "household_weight",
                    "weight_mapping": "household_to_person",
                    "seed": SALSAC_QRF_SEED,
                    "n_estimators": SALSAC_QRF_ESTIMATORS,
                    "clamp_minimum": 0,
                    "preserve_asked_rows": True,
                    "cache": False,
                },
            ),
            (
                "convert_donors_to_target_stock",
                {
                    "resource": "salary_sacrifice_anchor.json",
                    "target": int(SALSAC_STAGE_TARGET),
                    "donor_pool": (
                        "employee_pension_contributions > 0 and "
                        "pension_contributions_via_salary_sacrifice == 0 and "
                        "employment_income > 0"
                    ),
                    "rate_cap": SALSAC_RATE_CAP,
                    "move": (
                        "full employee_pension_contributions to "
                        "pension_contributions_via_salary_sacrifice; source zeroed"
                    ),
                    "seed": SALSAC_CONVERSION_SEED,
                    "salt": SALSAC_CONVERSION_SALT,
                    "receipt": "weighted_headcount",
                    "reason": SALSAC_MASS_CHANGE_REASON,
                },
            ),
        ),
    )
    hmrc = anchor.get("hmrc_anchor", {})
    derived = anchor.get("derived", {})
    checks = {
        "hmrc_anchor.total_users": (hmrc.get("total_users"), SALSAC_HMRC_ANCHOR),
        "hmrc_anchor.above_2000": (
            hmrc.get("above_2000"),
            SALSAC_ABOVE_2000_ANCHOR,
        ),
        "hmrc_anchor.below_2000": (
            hmrc.get("below_2000"),
            SALSAC_BELOW_2000_ANCHOR,
        ),
        "derived.stage_target": (derived.get("stage_target"), SALSAC_STAGE_TARGET),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"Salary-sacrifice resource {label} drifted: expected "
                f"{expected!r}, got {actual!r}."
            )
    ratio = float(derived.get("staging_ratio", np.nan))
    if not np.isclose(ratio, SALSAC_STAGING_RATIO, rtol=0.0, atol=1e-12):
        raise ValueError("Salary-sacrifice resource staging ratio drifted.")
