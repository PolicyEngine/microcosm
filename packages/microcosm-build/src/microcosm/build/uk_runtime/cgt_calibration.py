"""Materialize the declared UK capital gains constraint frame.

``uk_runtime.fiscal_targets`` declares two HMRC capital gains facts. This
module prepares the person-level measure columns those facts compile against
and assembles the constraint :class:`~microcosm.frame.Frame`, following the
shape of :mod:`microcosm.build.uk_runtime.hmrc_calibration`.

Unlike the HMRC SPI surface, no simulation is needed: ``capital_gains`` is a
persisted person-level input on the UK national frame, so the measures read
straight off its person table.

The CGT taxpayer indicator is ``capital_gains > annual_exempt_amount``. The AEA
is policy-dependent — it moved £12,300 -> £6,000 -> £3,000 across 2022-23 to
2024-25 — so the default is derived from the frame's time period through the
explicit :data:`UK_CGT_ANNUAL_EXEMPT_AMOUNTS` mapping and an unmapped period
raises. Silently defaulting would change what "CGT taxpayer" means without
changing the declared target value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.fiscal_targets import (
    UK_CGT_REQUIRED_COLUMNS,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_time_period,
)
from microcosm.calibrate import TargetRegistry
from microcosm.frame import EntitySchema, Frame, Weights

__all__ = [
    "UK_CGT_ANNUAL_EXEMPT_AMOUNTS",
    "UK_CGT_GAINS_AMOUNT_COLUMN",
    "UK_CGT_SOURCE_COLUMN",
    "UK_CGT_TAXPAYER_COUNT_COLUMN",
    "UKCGTTargetMaterialization",
    "materialize_uk_cgt_calibration_frame",
    "uk_cgt_annual_exempt_amount",
]

#: The persisted person-level input the measures are derived from.
UK_CGT_SOURCE_COLUMN = "capital_gains"

UK_CGT_GAINS_AMOUNT_COLUMN = "uk_cgt_measure_gains_amount"
UK_CGT_TAXPAYER_COUNT_COLUMN = "uk_cgt_measure_taxpayer_count"

#: CGT annual exempt amount by build period (tax year starting in that year).
#:
#: HMRC Capital Gains Tax rates and allowances:
#: https://www.gov.uk/government/publications/rates-and-allowances-capital-gains-tax
UK_CGT_ANNUAL_EXEMPT_AMOUNTS: dict[str, float] = {
    "2022": 12_300.0,
    "2023": 6_000.0,
    "2024": 3_000.0,
    "2025": 3_000.0,
    "2026": 3_000.0,
}


@dataclass(frozen=True)
class UKCGTTargetMaterialization:
    """Person-level CGT constraint frame and the declared HMRC facts."""

    frame: Frame
    registry: TargetRegistry
    annual_exempt_amount: float
    taxpayer_rows: int
    minimum_positive_support_rows: int


def uk_cgt_annual_exempt_amount(time_period: int | str) -> float:
    """Return the CGT annual exempt amount in force for ``time_period``.

    Raises:
        ValueError: If the period has no reviewed AEA. A wrong threshold
            silently changes what the declared taxpayer-count fact means, so
            an unmapped period is refused rather than defaulted.
    """

    period = str(time_period)
    try:
        return UK_CGT_ANNUAL_EXEMPT_AMOUNTS[period]
    except KeyError:
        raise ValueError(
            f"No reviewed CGT annual exempt amount for period {period!r}; "
            "the AEA is policy-dependent (£12,300 -> £6,000 -> £3,000 across "
            "2022-23 to 2024-25) and must be declared explicitly. Known "
            f"periods: {sorted(UK_CGT_ANNUAL_EXEMPT_AMOUNTS)}."
        ) from None


def materialize_uk_cgt_calibration_frame(
    national_frame: Frame,
    *,
    registry: TargetRegistry,
    annual_exempt_amount: float | None = None,
) -> UKCGTTargetMaterialization:
    """Prepare the CGT measure columns and assemble the constraint frame.

    Args:
        national_frame: UK national frame carrying the persisted person-level
            ``capital_gains`` input and strictly positive household weights.
        registry: CGT targets compiled from Ledger references.
        annual_exempt_amount: Override for the CGT threshold. Defaults to the
            reviewed AEA for the frame's time period.

    Returns:
        The frame, the two declared facts as a :class:`TargetRegistry`, and
        support diagnostics.

    Raises:
        ValueError: If ``capital_gains`` is absent or non-finite, if any
            household prior weight is not strictly positive, or if either
            target has no strictly positive-mass support.
    """

    if annual_exempt_amount is None:
        annual_exempt_amount = uk_cgt_annual_exempt_amount(
            uk_time_period(national_frame)
        )
    if not np.isfinite(annual_exempt_amount) or annual_exempt_amount < 0.0:
        raise ValueError(
            "CGT annual exempt amount must be finite and non-negative; got "
            f"{annual_exempt_amount!r}."
        )

    person = national_frame.table("person")
    household = national_frame.table("household")
    if UK_CGT_SOURCE_COLUMN not in person:
        raise ValueError(
            "UK CGT target materialization requires the person-level "
            f"{UK_CGT_SOURCE_COLUMN!r} input column; it is absent from the "
            "dataset person table, and the declared HMRC capital gains facts "
            "cannot be compiled without it."
        )
    capital_gains = pd.to_numeric(
        person[UK_CGT_SOURCE_COLUMN], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(capital_gains).all():
        raise ValueError(
            f"UK CGT person {UK_CGT_SOURCE_COLUMN!r} values must be finite."
        )

    household_weight_by_id = pd.Series(
        national_frame.weights_for("household").values,
        index=household["household_id"].to_numpy(),
    )
    mapped_mass = person["person_household_id"].map(household_weight_by_id)
    if mapped_mass.isna().any() or not mapped_mass.gt(0.0).all():
        raise ValueError(
            "UK CGT calibration requires strictly positive prior household "
            "mass for every person row."
        )
    positive_mass = mapped_mass.to_numpy(dtype=float) > 0.0

    support = capital_gains > float(annual_exempt_amount)
    measure_values = {
        UK_CGT_TAXPAYER_COUNT_COLUMN: support.astype(float),
        UK_CGT_GAINS_AMOUNT_COLUMN: np.where(support, capital_gains, 0.0),
    }
    if set(measure_values) != set(UK_CGT_REQUIRED_COLUMNS):
        raise RuntimeError(  # pragma: no cover - guarded by the declared facts
            "UK CGT prepared columns diverged from the declared measures: "
            f"{sorted(measure_values)} != {sorted(UK_CGT_REQUIRED_COLUMNS)}."
        )
    registry_measure_values = _registry_measure_values(registry, measure_values)

    n_positive = int((support & positive_mass).sum())
    if n_positive == 0:
        raise ValueError(
            "UK CGT targets have no strictly positive-mass support; refusing "
            "to calibrate positive HMRC capital gains facts (£65.9bn, 378k "
            "taxpayers) to zero constraint rows. Check the annual exempt "
            f"amount ({annual_exempt_amount:,.0f}) against the gains input."
        )

    calibration_person = person[["person_id", "person_household_id"]].reset_index(
        drop=True
    )
    calibration_person = calibration_person.assign(
        **measure_values,
        **registry_measure_values,
    )
    frame = Frame(
        {
            "person": calibration_person,
            "household": household[["household_id"]].copy(),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                national_frame.weights_for("household").values,
                uk_household_weight_kind(national_frame),
            )
        },
        mass_log=national_frame.mass_log,
    )
    return UKCGTTargetMaterialization(
        frame=frame,
        registry=registry,
        annual_exempt_amount=float(annual_exempt_amount),
        taxpayer_rows=int(support.sum()),
        minimum_positive_support_rows=n_positive,
    )


def _registry_measure_values(
    registry: TargetRegistry,
    prepared_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Expose Ledger-compiled measure names as aliases to stable CGT columns."""

    aliases = {
        "hmrc.cgt.gains_total": UK_CGT_GAINS_AMOUNT_COLUMN,
        "hmrc.cgt.taxpayers_total": UK_CGT_TAXPAYER_COUNT_COLUMN,
    }
    values: dict[str, np.ndarray] = {}
    for spec in registry:
        prepared_name = aliases.get(spec.name)
        if prepared_name is None or spec.measure in prepared_values:
            continue
        values[spec.measure] = prepared_values[prepared_name]
    return values
