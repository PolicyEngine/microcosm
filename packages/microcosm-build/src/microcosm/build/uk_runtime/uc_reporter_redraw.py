"""Redraw reported Universal Credit at benefit-unit grain on SPI support.

The stage deliberately runs after the person-grain SPI income chain and before
``uc_capital_coherence``.  It uses one temporary PolicyEngine materialization
to screen benefit units to positive pre-take-up awards and to obtain the UC
child definition, fits the existing regime-gated QRF on screened canonical
FRS benefit units, and rewrites only SPI ``universal_credit_reported`` rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.spi_support import (
    BASE_FRS_SUPPORT_CHANNEL,
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
    support_clone_index_column,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _boolean_values,
    _household_to_benunit_weights,
)
from microcosm.fit.qrf import RegimeGatedQRF
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

UC_REPORTER_REDRAW_STAGE_NAME = "uc_reporter_redraw"
UC_REPORTER_REDRAW_OUTPUT = "universal_credit_reported"
UC_REPORTER_REDRAW_SEED = 44
UC_REPORTER_SCREEN_VARIABLES = (
    "uc_maximum_amount",
    "uc_income_reduction",
    "is_child_or_qualifying_young_person_for_universal_credit",
)
UC_REPORTER_AGGREGATES = {
    "uc_child_count": "is_child_or_qualifying_young_person_for_universal_credit",
    "universal_credit_reported_amount": "universal_credit_reported",
}
UC_REPORTER_PREDICTORS = (
    "is_married",
    "uc_child_band",
    "benunit_employment_income",
    "benunit_self_employment_income",
    "claimant_earnings",
    "partner_earnings",
    "benunit_investment_income",
    "claimant_age",
    "region",
)
UC_REPORTER_NUMERIC_PREDICTORS = tuple(
    predictor for predictor in UC_REPORTER_PREDICTORS if predictor != "region"
)
UC_REPORTER_TARGET = "universal_credit_reported_amount"
UC_REPORTER_TEMPORARY_DERIVED = {
    "uc_reported_capital": "frs_benunit_capital",
    "positive_pre_takeup_uc_award": (
        "max(0, uc_maximum_amount - uc_income_reduction) > 0"
    ),
}
UC_REPORTER_TRAINING_ROWS = (
    "base_frs_channel_not_capital_gains_clone_support_clone_0_screened"
)
UC_REPORTER_TARGET_ROWS = "spi_channel_screened"
UC_REPORTER_LANDING = "eldest_working_age_adult_lowest_person_id"
_HOUSEHOLD_CGT_CLONE_COLUMN = "household_is_capital_gains_clone"
_PERSON_INCOME_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
    "other_investment_income",
)


@dataclass(frozen=True)
class UKUCReporterRedrawResult:
    """Output frame and executed-effect receipt for the UC reporter redraw."""

    frame: Frame
    training_benunits: int
    screened_spi_benunits: int
    screen_failed_spi_benunits: int
    reporter_transitions: Mapping[str, Mapping[str, Mapping[str, int]]]

    def evidence(self) -> dict[str, object]:
        """Return JSON-safe stage evidence."""

        return {
            "stage": UC_REPORTER_REDRAW_STAGE_NAME,
            "seed": UC_REPORTER_REDRAW_SEED,
            "training_benunits": self.training_benunits,
            "screened_spi_benunits": self.screened_spi_benunits,
            "screen_failed_spi_benunits": self.screen_failed_spi_benunits,
            "reporter_transitions": {
                channel: {
                    family_type: dict(counts)
                    for family_type, counts in families.items()
                }
                for channel, families in self.reporter_transitions.items()
            },
        }


@dataclass(frozen=True)
class UKUCReporterRedrawStageTransform:
    """Whole-stage callable for benefit-unit-grain SPI UC reporter redraw."""

    stage: SourceStageSpec
    engine: object
    qrf_factory: Callable[..., Any] = RegimeGatedQRF
    last_result: UKUCReporterRedrawResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = redraw_spi_reported_uc(
            frame,
            engine=self.engine,
            qrf_factory=self.qrf_factory,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return the completed stage's screen and reporter receipt."""

        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def redraw_spi_reported_uc(
    frame: Frame,
    *,
    engine: object,
    qrf_factory: Callable[..., Any] = RegimeGatedQRF,
) -> UKUCReporterRedrawResult:
    """Redraw SPI reported UC on the screened benefit-unit domain."""

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    _require_columns(
        person,
        (
            "person_id",
            "person_benunit_id",
            "person_household_id",
            "age",
            UC_REPORTER_REDRAW_OUTPUT,
            *_PERSON_INCOME_COLUMNS,
        ),
        label="person",
    )
    _require_columns(
        benunit,
        (
            "benunit_id",
            "frs_benunit_capital",
            "is_married",
            support_channel_column("benunit"),
            support_clone_index_column("benunit"),
        ),
        label="benunit",
    )
    _require_columns(
        household,
        ("household_id", "region"),
        label="household",
    )

    materialized = _materialize_screen_inputs(
        frame,
        engine=engine,
        person=person,
        benunit=benunit,
        household=household,
    )
    screen = _positive_pre_takeup_award_screen(
        materialized["uc_maximum_amount"],
        materialized["uc_income_reduction"],
        expected=len(benunit),
    )
    uc_child = _strict_materialized_bool(
        materialized[
            "is_child_or_qualifying_young_person_for_universal_credit"
        ],
        expected=len(person),
        label="UC child flag",
    )
    claimant_rows = _claimant_rows(person, uc_child=uc_child)
    predictors = _benefit_unit_predictors(
        person,
        benunit,
        household,
        uc_child=uc_child,
        claimant_rows=claimant_rows,
    )
    reported_before = _benefit_unit_reporter_amounts(person, benunit)
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=frame.weights_for("household").values,
    )
    channel = benunit[support_channel_column("benunit")].astype(str)
    base = channel.eq(BASE_FRS_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    spi = channel.eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy(dtype=bool)
    if np.any(~(base | spi)):
        raise ValueError("UC reporter redraw requires only FRS and SPI channels.")
    support_clone = _nonnegative_integer_values(
        benunit[support_clone_index_column("benunit")],
        label=support_clone_index_column("benunit"),
    )
    is_cgt_clone = _household_cgt_clone_by_benunit(
        person,
        benunit,
        household,
    )
    training = base & ~is_cgt_clone & (support_clone == 0) & screen
    target = spi & screen
    if not training.any():
        raise ValueError("UC reporter redraw has no screened canonical FRS training rows.")
    if float(weights[training].sum()) <= 0.0:
        raise ValueError("UC reporter redraw training weights must have positive mass.")

    model_table = predictors.loc[training].copy()
    model_table[UC_REPORTER_TARGET] = reported_before[training]
    target_table = predictors.loc[target].copy()
    encoded_train, encoded_target = _encode_predictor_pair(model_table, target_table)
    model_predictors = [
        column for column in encoded_train.columns if column != UC_REPORTER_TARGET
    ]
    fitted = qrf_factory(seed=UC_REPORTER_REDRAW_SEED).fit(
        encoded_train,
        model_predictors,
        [UC_REPORTER_TARGET],
        weights=weights[training],
    )
    draws = np.zeros(len(benunit), dtype=np.float64)
    if target.any():
        predicted = fitted.predict(encoded_target)
        if list(predicted.columns) != [UC_REPORTER_TARGET]:
            raise ValueError(
                "UC reporter redraw model must return exactly the declared target."
            )
        target_draws = pd.to_numeric(
            predicted[UC_REPORTER_TARGET], errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.isfinite(target_draws).all() or (target_draws < 0.0).any():
            raise ValueError("UC reporter redraw produced invalid reported amounts.")
        draws[target] = target_draws

    _land_spi_draws(
        person,
        benunit,
        spi=spi,
        claimant_rows=claimant_rows,
        draws=draws,
    )
    reported_after = _benefit_unit_reporter_amounts(person, benunit)
    if np.any(reported_after[spi & ~screen] != 0.0):
        raise RuntimeError("Screen-failing SPI benefit units retained reported UC.")
    transitions = _reporter_transition_receipt(
        benunit,
        before=reported_before > 0.0,
        after=reported_after > 0.0,
        uc_child=uc_child,
        person=person,
    )

    result_frame = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result_frame)
    return UKUCReporterRedrawResult(
        frame=result_frame,
        training_benunits=int(training.sum()),
        screened_spi_benunits=int(target.sum()),
        screen_failed_spi_benunits=int((spi & ~screen).sum()),
        reporter_transitions=transitions,
    )


def _materialize_screen_inputs(
    frame: Frame,
    *,
    engine: object,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> Mapping[str, np.ndarray]:
    temporary_benunit = benunit.copy()
    temporary_benunit["uc_reported_capital"] = pd.to_numeric(
        temporary_benunit["frs_benunit_capital"], errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    temporary = uk_national_frame(
        person=person,
        benunit=temporary_benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    materialized = engine.materialize(
        temporary,
        list(UC_REPORTER_SCREEN_VARIABLES),
        uk_time_period(frame),
    )
    missing = sorted(set(UC_REPORTER_SCREEN_VARIABLES) - set(materialized))
    if missing:
        raise ValueError(f"UC reporter redraw engine outputs missing: {missing}.")
    return materialized


def _positive_pre_takeup_award_screen(
    maximum: Any,
    reduction: Any,
    *,
    expected: int,
) -> np.ndarray:
    maximum_values = np.asarray(maximum, dtype=np.float64)
    reduction_values = np.asarray(reduction, dtype=np.float64)
    if maximum_values.shape != (expected,) or reduction_values.shape != (expected,):
        raise ValueError("UC award screen outputs must align to the benunit table.")
    if not np.isfinite(maximum_values).all() or not np.isfinite(reduction_values).all():
        raise ValueError("UC award screen outputs must be finite.")
    return np.maximum(0.0, maximum_values - reduction_values) > 0.0


def _strict_materialized_bool(values: Any, *, expected: int, label: str) -> np.ndarray:
    series = pd.Series(np.asarray(values))
    if len(series) != expected:
        raise ValueError(f"{label} must align to the person table.")
    return _boolean_values(series)


def _claimant_rows(person: pd.DataFrame, *, uc_child: np.ndarray) -> np.ndarray:
    age = _finite_numeric(person["age"], label="person.age")
    person_id = person["person_id"].to_numpy()
    adult = ~uc_child
    working_age = adult & (age < 66.0)
    claimant = np.zeros(len(person), dtype=bool)
    for _, positions in person.groupby("person_benunit_id", sort=False).indices.items():
        rows = np.asarray(positions, dtype=np.int64)
        candidates = rows[working_age[rows]]
        if not len(candidates):
            candidates = rows[adult[rows]]
        if not len(candidates):
            raise ValueError("Every benefit unit must contain an adult claimant candidate.")
        order = np.lexsort((person_id[candidates], -age[candidates]))
        claimant[candidates[order[0]]] = True
    return claimant


def _benefit_unit_predictors(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    uc_child: np.ndarray,
    claimant_rows: np.ndarray,
) -> pd.DataFrame:
    person_ids = person["person_benunit_id"]
    benunit_index = pd.Index(benunit["benunit_id"], name="benunit_id")
    numeric = {
        column: _finite_numeric(person[column], label=f"person.{column}")
        for column in ("age", *_PERSON_INCOME_COLUMNS)
    }
    person_numeric = pd.DataFrame(numeric, index=person.index)
    person_numeric["person_benunit_id"] = person_ids.to_numpy()
    grouped = person_numeric.groupby("person_benunit_id", sort=False)
    employment = grouped["employment_income"].sum().reindex(benunit_index)
    self_employment = grouped["self_employment_income"].sum().reindex(benunit_index)
    investment = grouped[
        [
            "savings_interest_income",
            "dividend_income",
            "property_income",
            "other_investment_income",
        ]
    ].sum().sum(axis=1).reindex(benunit_index)
    child_count = (
        pd.Series(uc_child.astype(np.int8), index=person.index)
        .groupby(person_ids, sort=False)
        .sum()
        .reindex(benunit_index)
    )
    claimant = person.loc[claimant_rows, ["person_benunit_id"]].copy()
    claimant["claimant_age"] = numeric["age"][claimant_rows]
    claimant["claimant_earnings"] = (
        numeric["employment_income"][claimant_rows]
        + numeric["self_employment_income"][claimant_rows]
    )
    claimant = claimant.set_index("person_benunit_id").reindex(benunit_index)
    adult_earnings = np.where(
        uc_child,
        0.0,
        numeric["employment_income"] + numeric["self_employment_income"],
    )
    adult_earnings_by_benunit = (
        pd.Series(adult_earnings, index=person.index)
        .groupby(person_ids, sort=False)
        .sum()
        .reindex(benunit_index)
    )
    region = _household_values_by_benunit(
        person,
        benunit,
        household,
        column="region",
    )
    result = pd.DataFrame(index=benunit.index)
    result["is_married"] = _boolean_values(benunit["is_married"]).astype(float)
    result["uc_child_band"] = np.minimum(child_count.to_numpy(dtype=float), 3.0)
    result["benunit_employment_income"] = employment.to_numpy(dtype=float)
    result["benunit_self_employment_income"] = self_employment.to_numpy(dtype=float)
    result["claimant_earnings"] = claimant["claimant_earnings"].to_numpy(dtype=float)
    result["partner_earnings"] = np.maximum(
        adult_earnings_by_benunit.to_numpy(dtype=float)
        - result["claimant_earnings"].to_numpy(dtype=float),
        0.0,
    )
    result["benunit_investment_income"] = investment.to_numpy(dtype=float)
    result["claimant_age"] = claimant["claimant_age"].to_numpy(dtype=float)
    result["region"] = region.to_numpy()
    if result[list(UC_REPORTER_NUMERIC_PREDICTORS)].isna().any().any():
        raise ValueError("UC reporter redraw numeric predictors contain missing values.")
    if result["region"].isna().any():
        raise ValueError("UC reporter redraw region predictor contains missing values.")
    return result.loc[:, list(UC_REPORTER_PREDICTORS)]


def _benefit_unit_reporter_amounts(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
) -> np.ndarray:
    amounts = pd.to_numeric(
        person[UC_REPORTER_REDRAW_OUTPUT], errors="coerce"
    ).fillna(0.0)
    if (amounts < 0.0).any():
        raise ValueError("universal_credit_reported must be nonnegative.")
    by_benunit = amounts.groupby(person["person_benunit_id"], sort=False).sum()
    mapped = benunit["benunit_id"].map(by_benunit).fillna(0.0)
    return mapped.to_numpy(dtype=np.float64)


def _encode_predictor_pair(
    train: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_values = train[UC_REPORTER_TARGET].copy()
    train_predictors = train.drop(columns=[UC_REPORTER_TARGET])
    combined = pd.concat(
        [train_predictors.reset_index(drop=True), target.reset_index(drop=True)],
        ignore_index=True,
    )
    encoded = pd.get_dummies(
        combined,
        columns=["region"],
        drop_first=False,
        dtype=float,
    )
    encoded = encoded.reindex(sorted(encoded.columns), axis=1)
    train_encoded = encoded.iloc[: len(train)].copy()
    train_encoded.index = train.index
    train_encoded[UC_REPORTER_TARGET] = target_values.to_numpy(dtype=float)
    target_encoded = encoded.iloc[len(train) :].copy()
    target_encoded.index = target.index
    for label, values in (
        ("training", train_encoded),
        ("target", target_encoded),
    ):
        numeric = values.to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"Encoded UC reporter {label} predictors are not finite.")
    return train_encoded, target_encoded


def _land_spi_draws(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    *,
    spi: np.ndarray,
    claimant_rows: np.ndarray,
    draws: np.ndarray,
) -> None:
    spi_ids = set(benunit.loc[spi, "benunit_id"])
    spi_people = person["person_benunit_id"].isin(spi_ids).to_numpy(dtype=bool)
    person.loc[spi_people, UC_REPORTER_REDRAW_OUTPUT] = 0.0
    draw_by_benunit = pd.Series(draws, index=benunit["benunit_id"])
    claimant_draws = person.loc[claimant_rows, "person_benunit_id"].map(
        draw_by_benunit
    )
    claimant_spi = claimant_rows & spi_people
    person.loc[claimant_spi, UC_REPORTER_REDRAW_OUTPUT] = claimant_draws.loc[
        claimant_spi[claimant_rows]
    ].to_numpy(dtype=float)


def _reporter_transition_receipt(
    benunit: pd.DataFrame,
    *,
    before: np.ndarray,
    after: np.ndarray,
    uc_child: np.ndarray,
    person: pd.DataFrame,
) -> dict[str, dict[str, dict[str, int]]]:
    child_count = (
        pd.Series(uc_child.astype(np.int8), index=person.index)
        .groupby(person["person_benunit_id"], sort=False)
        .sum()
    )
    children = benunit["benunit_id"].map(child_count).fillna(0).to_numpy(dtype=int)
    couple = _boolean_values(benunit["is_married"])
    family_types = np.where(
        couple,
        np.where(children > 0, "couple_with_children", "couple_no_children"),
        np.where(children > 0, "single_with_children", "single_no_children"),
    )
    channels = benunit[support_channel_column("benunit")].astype(str).to_numpy()
    receipt: dict[str, dict[str, dict[str, int]]] = {}
    for channel in sorted(set(channels)):
        receipt[channel] = {}
        for family_type in sorted(set(family_types[channels == channel])):
            cell = (channels == channel) & (family_types == family_type)
            receipt[channel][family_type] = {
                "promoted": int((cell & ~before & after).sum()),
                "demoted": int((cell & before & ~after).sum()),
                "held_reporter": int((cell & before & after).sum()),
                "held_nonreporter": int((cell & ~before & ~after).sum()),
            }
    return receipt


def _household_cgt_clone_by_benunit(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> np.ndarray:
    if _HOUSEHOLD_CGT_CLONE_COLUMN not in household:
        return np.zeros(len(benunit), dtype=bool)
    values = _household_values_by_benunit(
        person,
        benunit,
        household,
        column=_HOUSEHOLD_CGT_CLONE_COLUMN,
    )
    values.name = _HOUSEHOLD_CGT_CLONE_COLUMN
    return _boolean_values(values)


def _household_values_by_benunit(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    column: str,
) -> pd.Series:
    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    counts = placements.groupby("person_benunit_id", sort=False)[
        "person_household_id"
    ].nunique()
    if (counts != 1).any():
        raise ValueError("Every benefit unit must map to exactly one household.")
    household_by_benunit = placements.set_index("person_benunit_id")[
        "person_household_id"
    ]
    values_by_household = household.set_index("household_id")[column]
    mapped = benunit["benunit_id"].map(household_by_benunit).map(values_by_household)
    if mapped.isna().any():
        raise ValueError(f"Household {column} does not cover every benefit unit.")
    return mapped


def _nonnegative_integer_values(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = _finite_numeric(values, label=label)
    if (numeric < 0.0).any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{label} must contain nonnegative integers.")
    return numeric.astype(np.int64)


def _finite_numeric(values: pd.Series, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must be finite numeric.")
    return numeric


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    if stage.stage != UC_REPORTER_REDRAW_STAGE_NAME:
        raise ValueError(
            f"UC reporter redraw received stage {stage.stage!r}, expected "
            f"{UC_REPORTER_REDRAW_STAGE_NAME!r}."
        )
    kinds = [operation.kind for operation in stage.operations]
    expected_kinds = [
        "derive",
        "materialize_rules_engine_predictors",
        "aggregate_person_to_benunit",
        "redraw_spi_reported_uc",
    ]
    if kinds != expected_kinds:
        raise ValueError(
            "uc_reporter_redraw operation order drifted: "
            f"expected {expected_kinds}, got {kinds}."
        )
    derive, materialize, aggregate, redraw = stage.operations
    expected = {
        "derive": UC_REPORTER_TEMPORARY_DERIVED,
        "materialize": {
            "predictors": list(UC_REPORTER_SCREEN_VARIABLES),
            "consumed_only": True,
        },
        "aggregate": {
            "method": "sum",
            "consumed_only": True,
            "aggregates": UC_REPORTER_AGGREGATES,
        },
        "redraw": {
            "output": UC_REPORTER_REDRAW_OUTPUT,
            "training_rows": UC_REPORTER_TRAINING_ROWS,
            "rows": UC_REPORTER_TARGET_ROWS,
            "screen": "positive_pre_takeup_uc_award",
            "predictors": list(UC_REPORTER_PREDICTORS),
            "categorical_predictors": ["region"],
            "target": UC_REPORTER_TARGET,
            "weight_mapping": "household_to_benunit",
            "fit": "regime_gated_qrf",
            "landing": UC_REPORTER_LANDING,
            "seed": UC_REPORTER_REDRAW_SEED,
        },
    }
    actual = {
        "derive": derive.parameters.get("derived"),
        "materialize": dict(materialize.parameters),
        "aggregate": dict(aggregate.parameters),
        "redraw": dict(redraw.parameters),
    }
    if actual != expected:
        raise ValueError(
            "uc_reporter_redraw parameters drifted: "
            f"expected {expected}, got {actual}."
        )
    if stage.outputs != () or stage.rewrites != (UC_REPORTER_REDRAW_OUTPUT,):
        raise ValueError(
            "uc_reporter_redraw must declare no new outputs and exactly the "
            "universal_credit_reported rewrite."
        )


def _require_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    label: str,
) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"UC reporter redraw {label} columns missing: {missing}.")
