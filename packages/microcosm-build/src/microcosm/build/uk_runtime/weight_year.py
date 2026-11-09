"""The year a UK release's household weights are written at (microcosm#796 A19).

policyengine-uk projects a single-year dataset forward. From the dataset's own
year it multiplies ``household_weight`` by one plus the ONS population growth
of each later year (``policyengine_uk/data/uprating_indices.yaml`` lists the
column under ``gov.economic_assumptions.yoy_growth.ons.population``; no other
weight is stored, and the person and benefit-unit weights are formulas of it).

The UK release is stamped with the survey year and its weights are solved at
the calibration year. Written as solved, they read high at the calibration
year by the growth between the two, which the ``dataset_projection`` block of
``uk/hmrc_uprating_engine_pins.json`` pins for each engine lock. uk-data#538 is the
incumbent's version of the same mechanism, with the opposite sign.

So the release writes the solved weights divided by the engine's own factor,
and the engine's projection then returns the solved weights at the solve year.
The division happens on a derived frame at the release write only. The solve,
its gates and the engine scratch file keep the solved weights.

The growth values are vendored beside the uprating pins
(``uk/hmrc_uprating_engine_pins.json``, block ``dataset_projection``) by
``tools/pin_uk_uprating_engine_values.py``, so nothing here imports the engine;
a ``requires_uk`` test holds them in lockstep with the installed engine.
"""

from __future__ import annotations

import numpy as np

from microcosm.build.uk_runtime.hmrc_uprating import (
    UK_ENGINE_PINS_RESOURCE,
    UK_ENGINE_PINS_TOOL,
    engine_parameter_pins,
)
from microcosm.build.uk_runtime.national_frame import uk_time_period
from microcosm.frame import Frame, MassChange, WeightKind, Weights

#: The engine parameter policyengine-uk projects ``household_weight`` by: the
#: year-on-year growth it applies entering each year after the dataset's own.
UK_WEIGHT_PROJECTION_PARAMETER = "gov.economic_assumptions.yoy_growth.ons.population"
UK_WEIGHT_PROJECTION_VARIABLE = "household_weight"
#: The pins block that carries the growth, keyed by the projected variable.
UK_WEIGHT_PROJECTION_PINS_BLOCK = "dataset_projection"

#: The mass-log reason of the division. It names the solve, not "calibration":
#: the solve's own evidence finds its record by that word, on the frame before
#: this one is derived.
UK_WEIGHT_YEAR_MASS_REASON = (
    "Release weights at the dataset year: the solved household weights divided "
    "by policyengine-uk's own projection of household_weight from the dataset "
    "year to the solve year, so the engine returns the solved weights at the "
    "solve year (microcosm#796 A19)."
)
#: The mass-log reason of the lift a reader applies to see a stored file as
#: the engine does at a later year.
UK_ENGINE_YEAR_MASS_REASON = (
    "Household weights as policyengine-uk projects them from the dataset year "
    "to the simulation year (microcosm#796 A19)."
)


def uk_weight_projection_growth(year: int) -> float:
    """The pinned growth the engine applies to ``household_weight`` entering ``year``."""

    block = engine_parameter_pins().get(UK_WEIGHT_PROJECTION_PINS_BLOCK) or {}
    entry = block.get(UK_WEIGHT_PROJECTION_VARIABLE) or {}
    growth = entry.get("growth") or {}
    instant = f"{int(year)}-01-01"
    if (
        entry.get("parameter") != UK_WEIGHT_PROJECTION_PARAMETER
        or instant not in growth
    ):
        raise ValueError(
            f"{UK_ENGINE_PINS_RESOURCE} does not pin {UK_WEIGHT_PROJECTION_PARAMETER!r} "
            f"at {instant} for {UK_WEIGHT_PROJECTION_VARIABLE}; re-run "
            f"{UK_ENGINE_PINS_TOOL} with the uk extra installed."
        )
    value = float(growth[instant])
    if not np.isfinite(value) or value <= -1.0:
        raise ValueError(
            f"{UK_ENGINE_PINS_RESOURCE}: {UK_WEIGHT_PROJECTION_PARAMETER!r} at "
            f"{instant} is {value!r}; a growth rate must be finite and above -1."
        )
    return value


def uk_weight_projection_factor(dataset_year: int, simulation_year: int) -> float:
    """What the engine multiplies ``household_weight`` by between the two years.

    One for the dataset's own year. The engine does not project backwards, so
    an earlier simulation year refuses.
    """

    dataset_year, simulation_year = int(dataset_year), int(simulation_year)
    if simulation_year < dataset_year:
        raise ValueError(
            f"policyengine-uk does not project a {dataset_year} dataset back to "
            f"{simulation_year}."
        )
    factor = 1.0
    for year in range(dataset_year + 1, simulation_year + 1):
        factor *= 1.0 + uk_weight_projection_growth(year)
    return factor


def uk_weights_are_at_dataset_year(frame: Frame) -> bool:
    """Whether the frame's household weights were already divided for release."""

    records = [record for record in frame.mass_log if record.entity == "household"]
    return bool(records) and records[-1].reason == UK_WEIGHT_YEAR_MASS_REASON


def uk_dataset_year_frame(frame: Frame, *, solve_year: int) -> Frame:
    """The release frame: solved household weights at the dataset's year.

    Idempotent: a frame that already carries the division is returned as it
    is. Only calibrated weights are divided, since the division undoes the
    engine's projection of a solve; anything else refuses.
    """

    if uk_weights_are_at_dataset_year(frame):
        return frame
    weights = frame.weights_for("household")
    if weights.kind is not WeightKind.CALIBRATED:
        raise ValueError(
            "Only solved household weights are written at the dataset year; "
            f"these are {weights.kind.value!r}."
        )
    factor = uk_weight_projection_factor(int(uk_time_period(frame)), solve_year)
    return frame.with_weights(
        "household",
        Weights(np.asarray(weights.values, dtype=np.float64) / factor, weights.kind),
        mass=MassChange(factor=1.0 / factor, reason=UK_WEIGHT_YEAR_MASS_REASON),
    )


def uk_engine_year_frame(frame: Frame, *, simulation_year: int) -> Frame:
    """The frame with the household weights the engine uses at ``simulation_year``.

    This is the engine's projection of the stored weights, whatever wrote
    them: for a release written at the dataset year it gives back the solved
    weights at the solve year, and for a file written as solved it shows the
    overstatement its users get. The dataset's own year returns the frame.
    """

    factor = uk_weight_projection_factor(int(uk_time_period(frame)), simulation_year)
    if factor == 1.0:
        return frame
    weights = frame.weights_for("household")
    return frame.with_weights(
        "household",
        Weights(np.asarray(weights.values, dtype=np.float64) * factor, weights.kind),
        mass=MassChange(factor=factor, reason=UK_ENGINE_YEAR_MASS_REASON),
    )


def uk_solve_year_frame(frame: Frame, *, solve_year: int) -> Frame:
    """A written frame at the weights it was solved with.

    The gates that certify a release stand on the solved weights. A release
    carries them divided, and the engine's projection to the solve year gives
    them back; a frame that was never divided is returned as it is, since its
    stored weights are the solved ones. A frame divided by anything but the
    engine's factor to ``solve_year`` refuses: the engine would not return
    its solve.
    """

    if not uk_weights_are_at_dataset_year(frame):
        return frame
    factor = uk_weight_projection_factor(int(uk_time_period(frame)), solve_year)
    record = [r for r in frame.mass_log if r.entity == "household"][-1]
    declared = record.declared_factor
    if declared is None or not np.isclose(
        float(declared) * factor, 1.0, rtol=1e-12, atol=0.0
    ):
        raise ValueError(
            "The frame's household weights were divided by a factor that is not "
            f"policyengine-uk's projection of {UK_WEIGHT_PROJECTION_VARIABLE} to "
            f"{int(solve_year)}; the engine would not return the solved weights "
            "there."
        )
    return uk_engine_year_frame(frame, simulation_year=solve_year)


__all__ = [
    "UK_ENGINE_YEAR_MASS_REASON",
    "UK_WEIGHT_PROJECTION_PARAMETER",
    "UK_WEIGHT_PROJECTION_PINS_BLOCK",
    "UK_WEIGHT_PROJECTION_VARIABLE",
    "UK_WEIGHT_YEAR_MASS_REASON",
    "uk_dataset_year_frame",
    "uk_engine_year_frame",
    "uk_solve_year_frame",
    "uk_weight_projection_factor",
    "uk_weight_projection_growth",
    "uk_weights_are_at_dataset_year",
]
