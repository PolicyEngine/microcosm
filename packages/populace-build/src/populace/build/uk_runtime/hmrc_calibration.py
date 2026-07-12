"""Materialize and calibrate the complete UK HMRC SPI income surface."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from populace.build.uk_runtime.hmrc_income import (
    HMRC_SPI_ASSESSABLE_INCOME_COLUMN,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_PUBLICATION_URL,
    HMRC_SPI_TARGET_RECORD_COUNT,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeTargetSet,
)
from populace.build.uk_runtime.national_build import UKNationalDataset
from populace.build.uk_runtime.spi_income import derive_hmrc_income_auxiliaries
from populace.build.uk_runtime.spi_support import (
    SPI_HMRC_EMPLOYED_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN,
    SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN,
)
from populace.calibrate import (
    CONSERVE_MASS,
    CalibrationResult,
    TargetRegistry,
    TargetSpec,
    calibrate,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

__all__ = [
    "DEFAULT_HMRC_CALIBRATION_EPOCHS",
    "DEFAULT_HMRC_CALIBRATION_LEARNING_RATE",
    "DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR",
    "DEFAULT_HMRC_MAX_WEIGHT_RATIO",
    "HMRC_ASSESSABLE_INCOME_COLUMN",
    "HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN",
    "HMRC_TAXPAYER_COLUMN",
    "UKHMRCIncomeCalibration",
    "UKHMRCTargetMaterialization",
    "calibrate_uk_hmrc_income",
    "materialize_uk_hmrc_calibration_frame",
]

HMRC_TAXPAYER_COLUMN = "hmrc_spi_taxpayer"
HMRC_ASSESSABLE_INCOME_COLUMN = HMRC_SPI_ASSESSABLE_INCOME_COLUMN
HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN = "hmrc_spi_taxable_savings_interest_income"
DEFAULT_HMRC_CALIBRATION_EPOCHS = 256
DEFAULT_HMRC_CALIBRATION_LEARNING_RATE = 0.02
DEFAULT_HMRC_MAX_WEIGHT_RATIO = 5.0
DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR = 0.05

_SIMULATED_PERSON_COLUMNS = (
    "person_id",
    "income_tax",
)


@dataclass(frozen=True)
class UKHMRCTargetMaterialization:
    """Complete person-level constraint frame and versioned HMRC facts."""

    frame: Frame
    registry: TargetRegistry
    source_targets: HMRCIncomeTargetSet
    taxpayer_rows: int
    minimum_positive_support_rows: int


@dataclass(frozen=True)
class UKHMRCIncomeCalibration:
    """Guarded HMRC calibration result and its release-quality diagnostics."""

    result: CalibrationResult
    registry: TargetRegistry
    maximum_abs_relative_error: float
    worst_target: str


def materialize_uk_hmrc_calibration_frame(
    dataset: UKNationalDataset,
    source_targets: HMRCIncomeTargetSet,
    *,
    simulation_factory: Callable[[Any], Any] | None = None,
) -> UKHMRCTargetMaterialization:
    """Build all 208 HMRC rows using the official taxpayer/band semantics.

    PolicyEngine-UK supplies the person-level ``income_tax`` taxpayer mask.
    Income-band placement derives ``TI = TEI + TII`` from the same post-source
    leaves on every FRS and SPI row. Table 3.6's published employment and state
    pension measures likewise use the explicit HMRC auxiliaries on both
    channels; PolicyEngine's narrower employment input and state-pension inputs
    remain unchanged.
    ``calculate_dataframe(..., map_to="person")`` retains PolicyEngine's entity
    mapping and MicroSeries weights while values align back to stable IDs.
    """

    _validate_materialization_inputs(dataset, source_targets)
    person = dataset.person.copy()
    household = dataset.household.copy()

    simulation_input = _uk_single_year_dataset(dataset)
    factory = simulation_factory or _default_simulation_factory
    simulation = factory(simulation_input)
    calculated = simulation.calculate_dataframe(
        list(_SIMULATED_PERSON_COLUMNS),
        period=dataset.time_period,
        map_to="person",
        use_weights=True,
    )
    simulated = pd.DataFrame(calculated)[list(_SIMULATED_PERSON_COLUMNS)].copy()
    simulated = _strict_simulated_person_values(simulated, person)

    simulated_by_id = simulated.set_index("person_id")
    person_ids = person["person_id"]
    person["income_tax"] = person_ids.map(simulated_by_id["income_tax"]).to_numpy()
    person[HMRC_TAXPAYER_COLUMN] = person["income_tax"] > 0.0
    # Recompute every HMRC aggregate from normalized leaves at the target
    # boundary. A caller cannot smuggle a channel-specific precomputed proxy
    # into Table 3.6 or the band assignment auxiliary.
    person = derive_hmrc_income_auxiliaries(person)

    component_values: dict[str, np.ndarray] = {}
    tax_free_values: np.ndarray | None = None
    for component in HMRC_SPI_INCOME_COMPONENTS:
        if component == "employment_income":
            source_column = SPI_HMRC_EMPLOYED_INCOME_COLUMN
        elif component == "state_pension":
            source_column = SPI_HMRC_STATE_PENSION_INCOME_COLUMN
        else:
            source_column = component
        if source_column not in person:
            raise ValueError(
                "HMRC target materialization cannot silently narrow missing "
                f"component {component!r}; required source column "
                f"{source_column!r} is unavailable."
            )
        values = pd.to_numeric(person[source_column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"HMRC component {component!r} must be finite.")
        if component == "savings_interest_income":
            if "tax_free_savings_income" not in person:
                raise ValueError(
                    "HMRC savings-interest materialization requires "
                    "tax_free_savings_income so Table 3.7 excludes tax-exempt "
                    "interest."
                )
            tax_free = pd.to_numeric(
                person["tax_free_savings_income"], errors="coerce"
            ).to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(tax_free).all() or (tax_free < 0.0).any():
                raise ValueError(
                    "HMRC tax-free savings interest must be finite and non-negative."
                )
            if (values < tax_free).any():
                raise ValueError(
                    "HMRC taxable-savings bridge requires PolicyEngine gross "
                    "savings_interest_income to be at least "
                    "tax_free_savings_income on every person row."
                )
            # PolicyEngine's persisted input is gross savings interest, while
            # HMRC Table 3.7 and its total-income bands exclude tax-exempt
            # interest. Keep the PE input intact and derive the taxable source
            # component explicitly for both the measure and band assignment.
            values = values - tax_free
            tax_free_values = tax_free
            person[HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN] = values
        component_values[component] = values

    if tax_free_values is None:  # pragma: no cover - fixed component contract
        raise RuntimeError("HMRC savings-interest bridge was not materialized.")
    auxiliary_values: dict[str, np.ndarray] = {}
    for column in (
        SPI_HMRC_OTHER_INCOME_COLUMN,
        SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    ):
        if column not in person:
            raise ValueError(
                "HMRC accounting crosswalk cannot silently narrow missing "
                f"constituent {column!r}."
            )
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"HMRC accounting constituent {column!r} is invalid.")
        auxiliary_values[column] = values

    total_earned = (
        component_values["employment_income"]
        + auxiliary_values[SPI_HMRC_OTHER_INCOME_COLUMN]
        + auxiliary_values[SPI_HMRC_STATE_PENSION_INCOME_COLUMN]
        + component_values["self_employment_income"]
        + component_values["private_pension_income"]
    )
    total_investment = (
        component_values["savings_interest_income"]
        + component_values["dividend_income"]
        + component_values["property_income"]
        + component_values["other_investment_income"]
    )
    assessable_income = total_earned + total_investment
    if not np.isfinite(assessable_income).all():
        raise ValueError("HMRC assessable component income must be finite.")
    person[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN] = total_earned
    person[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN] = total_investment
    person[HMRC_ASSESSABLE_INCOME_COLUMN] = assessable_income
    if not np.array_equal(
        person[HMRC_ASSESSABLE_INCOME_COLUMN].to_numpy(dtype=float),
        person[SPI_HMRC_TOTAL_EARNED_INCOME_COLUMN].to_numpy(dtype=float)
        + person[SPI_HMRC_TOTAL_INVESTMENT_INCOME_COLUMN].to_numpy(dtype=float),
    ):
        raise RuntimeError("HMRC TI must equal derived TEI + TII exactly.")

    positive_household_mass = household.set_index("household_id")["household_weight"]
    mapped_mass = person["person_household_id"].map(positive_household_mass)
    if mapped_mass.isna().any() or not mapped_mass.gt(0.0).all():
        raise ValueError(
            "HMRC calibration requires strictly positive prior household mass "
            "for every person row."
        )
    taxpayers = person[HMRC_TAXPAYER_COLUMN].to_numpy(dtype=bool)

    target_specs: list[TargetSpec] = []
    measure_payload: dict[str, pd.arrays.SparseArray] = {}
    support_rows: list[int] = []
    for index, target in enumerate(source_targets.targets):
        band = _band_mask(assessable_income, target)
        component = component_values[target.component]
        support = taxpayers & band & (component > 0.0)
        measure_column = f"hmrc_spi_measure_{index:03d}"
        if target.measure == "count":
            measure_values = support.astype(float)
        elif target.measure == "amount":
            measure_values = np.where(support, component, 0.0)
        else:  # pragma: no cover - guarded by the typed parser contract
            raise ValueError(f"Unknown HMRC measure {target.measure!r}.")
        positive_support = support & (mapped_mass.to_numpy(dtype=float) > 0.0)
        n_positive = int(positive_support.sum())
        if n_positive == 0:
            raise ValueError(
                f"HMRC target {target.name!r} has no strictly positive-mass "
                "support; refusing to calibrate a positive source target to a "
                "zero constraint row."
            )
        # Each person contributes to only one income band. Sparse columns keep
        # the full fail-closed 208-row surface tractable on the national file;
        # the calibrator densifies one constraint row at a time before writing
        # its CSR matrix.
        measure_payload[measure_column] = pd.arrays.SparseArray(
            measure_values,
            fill_value=0.0,
        )
        support_rows.append(n_positive)
        target_specs.append(_target_spec(target, measure_column))

    calibration_person = pd.concat(
        [
            person[
                [
                    "person_id",
                    "person_household_id",
                    HMRC_TAXPAYER_COLUMN,
                    HMRC_ASSESSABLE_INCOME_COLUMN,
                    HMRC_TAXABLE_SAVINGS_INTEREST_COLUMN,
                ]
            ].reset_index(drop=True),
            pd.DataFrame(measure_payload),
        ],
        axis=1,
    )
    calibration_household = household[["household_id"]].copy()
    frame = Frame(
        {
            "person": calibration_person,
            "household": calibration_household,
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                household["household_weight"].to_numpy(dtype=float),
                dataset.household_weight_kind,
            )
        },
        mass_log=dataset.mass_log,
    )
    registry = TargetRegistry(target_specs, country="uk")
    if len(registry) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise ValueError(
            "HMRC calibration registry must contain exactly "
            f"{HMRC_SPI_TARGET_RECORD_COUNT} facts; got {len(registry)}."
        )
    return UKHMRCTargetMaterialization(
        frame=frame,
        registry=registry,
        source_targets=source_targets,
        taxpayer_rows=int(taxpayers.sum()),
        minimum_positive_support_rows=min(support_rows),
    )


def calibrate_uk_hmrc_income(
    materialization: UKHMRCTargetMaterialization,
    *,
    epochs: int = DEFAULT_HMRC_CALIBRATION_EPOCHS,
    learning_rate: float = DEFAULT_HMRC_CALIBRATION_LEARNING_RATE,
    max_weight_ratio: float = DEFAULT_HMRC_MAX_WEIGHT_RATIO,
    maximum_abs_relative_error: float = DEFAULT_HMRC_MAX_ABS_RELATIVE_ERROR,
    seed: int = 42,
) -> UKHMRCIncomeCalibration:
    """Conservatively calibrate household weights and fail closed on fit."""

    if materialization.frame.resolve_weights("household").kind is not (
        WeightKind.IMPORTANCE
    ):
        raise ValueError("HMRC calibration requires importance-kind input weights.")
    if not math.isfinite(maximum_abs_relative_error) or not (
        0.0 <= maximum_abs_relative_error < 1.0
    ):
        raise ValueError("maximum_abs_relative_error must be finite and in [0, 1).")
    result = calibrate(
        materialization.frame,
        materialization.registry.to_target_set(),
        weight_entity="household",
        epochs=epochs,
        learning_rate=learning_rate,
        mass=CONSERVE_MASS,
        max_weight_ratio=max_weight_ratio,
        seed=seed,
    )
    if result.skipped:
        skipped = [item.target.row_name for item in result.skipped]
        raise RuntimeError(f"HMRC calibration skipped target(s): {skipped}.")
    if len(result.diagnostics) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise RuntimeError(
            "HMRC calibration diagnostics are incomplete: expected "
            f"{HMRC_SPI_TARGET_RECORD_COUNT}, got {len(result.diagnostics)}."
        )
    if not np.isfinite(result.weights).all() or not (result.weights > 0.0).all():
        raise RuntimeError("HMRC calibrated household weights must be positive/finite.")
    initial_total = float(np.asarray(result.initial_weights).sum())
    final_total = float(np.asarray(result.weights).sum())
    if not np.isclose(initial_total, final_total, rtol=1e-9, atol=0.0):
        raise RuntimeError(
            "HMRC calibration changed national household mass despite "
            f"mass='conserve': {initial_total} -> {final_total}."
        )
    relative_errors = np.asarray(
        [diagnostic.relative_error for diagnostic in result.diagnostics],
        dtype=float,
    )
    if not np.isfinite(relative_errors).all():
        raise RuntimeError("HMRC calibration diagnostics contain non-finite errors.")
    worst_index = int(np.argmax(np.abs(relative_errors)))
    worst_error = float(abs(relative_errors[worst_index]))
    worst_target = result.diagnostics[worst_index].name
    if worst_error > maximum_abs_relative_error:
        raise RuntimeError(
            f"HMRC calibration target {worst_target!r} has absolute relative "
            f"error {worst_error:.6g}, exceeding the reviewed "
            f"{maximum_abs_relative_error:.6g} release limit."
        )
    return UKHMRCIncomeCalibration(
        result=result,
        registry=materialization.registry,
        maximum_abs_relative_error=worst_error,
        worst_target=worst_target,
    )


def _validate_materialization_inputs(
    dataset: UKNationalDataset,
    source_targets: HMRCIncomeTargetSet,
) -> None:
    if dataset.household_weight_kind is not WeightKind.IMPORTANCE:
        raise ValueError(
            "HMRC target materialization requires rebuilt importance weights."
        )
    weights = pd.to_numeric(dataset.household["household_weight"], errors="coerce")
    if weights.isna().any() or not weights.gt(0.0).all():
        raise ValueError(
            "HMRC target materialization requires every household prior weight "
            "to be strictly positive."
        )
    if source_targets.source.build_period != dataset.time_period:
        raise ValueError(
            "HMRC target source period does not match the UK build period: "
            f"{source_targets.source.build_period!r} != {dataset.time_period!r}."
        )
    if len(source_targets.targets) != HMRC_SPI_TARGET_RECORD_COUNT:
        raise ValueError(
            "HMRC target source must contain the complete 208-record surface."
        )
    expected = {
        (lower, component, measure)
        for lower in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
        for component in HMRC_SPI_INCOME_COMPONENTS
        for measure in ("count", "amount")
    }
    actual = {
        (target.total_income_lower_bound, target.component, target.measure)
        for target in source_targets.targets
    }
    if actual != expected:
        raise ValueError(
            "HMRC target source is incomplete or contains unexpected records; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )


def _uk_single_year_dataset(dataset: UKNationalDataset) -> Any:
    try:
        from policyengine_uk.data import UKSingleYearDataset
    except ImportError as exc:  # pragma: no cover - runtime dependency diagnostic
        raise ImportError(
            "HMRC target materialization requires policyengine-uk."
        ) from exc
    return UKSingleYearDataset(
        person=dataset.person.copy(),
        benunit=dataset.benunit.copy(),
        household=dataset.household.copy(),
        fiscal_year=int(dataset.time_period),
    )


def _default_simulation_factory(dataset: Any) -> Any:
    try:
        from policyengine_uk import Microsimulation
    except ImportError as exc:  # pragma: no cover - runtime dependency diagnostic
        raise ImportError(
            "HMRC target materialization requires policyengine-uk."
        ) from exc
    return Microsimulation(dataset=dataset)


def _strict_simulated_person_values(
    simulated: pd.DataFrame,
    person: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(set(_SIMULATED_PERSON_COLUMNS) - set(simulated.columns))
    if missing:
        raise ValueError(f"PolicyEngine-UK omitted HMRC variable(s): {missing}.")
    if simulated["person_id"].isna().any() or simulated["person_id"].duplicated().any():
        raise ValueError("PolicyEngine-UK HMRC person IDs must be complete and unique.")
    expected_ids = set(person["person_id"])
    actual_ids = set(simulated["person_id"])
    if actual_ids != expected_ids:
        raise ValueError(
            "PolicyEngine-UK HMRC person surface does not align to the input; "
            f"missing={list(expected_ids - actual_ids)[:5]}, "
            f"unexpected={list(actual_ids - expected_ids)[:5]}."
        )
    for column in ("income_tax",):
        values = pd.to_numeric(simulated[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"PolicyEngine-UK {column} must be finite.")
        simulated[column] = values
    return simulated


def _band_mask(
    income: np.ndarray,
    target: HMRCIncomeBandTargetRecord,
) -> np.ndarray:
    mask = income >= float(target.total_income_lower_bound)
    if target.total_income_upper_bound is not None:
        mask &= income < float(target.total_income_upper_bound)
    return mask


def _target_spec(
    target: HMRCIncomeBandTargetRecord,
    measure_column: str,
) -> TargetSpec:
    return TargetSpec(
        name=target.name,
        entity="person",
        value=target.value,
        measure=measure_column,
        period=target.period,
        source=(
            f"HMRC Personal Incomes 2023-24 Tables 3.6/3.7; {HMRC_SPI_PUBLICATION_URL}"
        ),
        family="hmrc_spi_income",
        notes=(
            "Taxpayers with income_tax > 0; band assignment derives TI as "
            "TEI + TII from the declared HMRC auxiliary crosswalk on every "
            "FRS and SPI row."
        ),
        metadata={
            "component": target.component,
            "measure": target.measure,
            "unit": target.unit,
            "income_lower_bound": str(target.total_income_lower_bound),
            "income_upper_bound": (
                "inf"
                if target.total_income_upper_bound is None
                else str(target.total_income_upper_bound)
            ),
            "taxpayer_mask": HMRC_TAXPAYER_COLUMN,
            "band_measure": HMRC_ASSESSABLE_INCOME_COLUMN,
        },
    )
