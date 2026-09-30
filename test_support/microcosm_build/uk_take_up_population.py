"""Engine-free stand-ins for the UK take-up population (is_WA_adult)."""

from __future__ import annotations

import numpy as np

from microcosm.build.uk_runtime.frs_take_up import (
    UK_TAKE_UP_ENGINE_PREDICTORS,
    UK_UC_WORKING_AGE_ADULT,
    UKTakeUpPopulation,
)


def whole_age_working_age_adult(age, *, state_pension_age: int = 66) -> np.ndarray:
    """policyengine-uk's is_WA_adult where State Pension age is one whole age.

    Until 2025-26 State Pension age is 66 for everyone who can reach it in a
    release, so the engine's answer is ``18 <= age < 66`` for every person.
    """

    values = np.asarray(age, dtype=float)
    return (values >= 18) & (values < state_pension_age)


def population_from_ages(person, *, period: str = "2024") -> UKTakeUpPopulation:
    """The take-up population a pre-2026 engine gives for ``person``."""

    return UKTakeUpPopulation(
        working_age_adult=whole_age_working_age_adult(person["age"]),
        period=period,
        source="test",
    )


class WorkingAgeStubEngine:
    """Answers the take-up stage's engine read without policyengine-uk.

    By default each person is a working-age adult when ``18 <= age < 66``;
    pass ``working_age_adult`` to set the engine's per-row answer directly,
    as a split State Pension age cohort needs. Calls are recorded.
    """

    country = "uk"

    def __init__(self, working_age_adult=None) -> None:
        self.working_age_adult = working_age_adult
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def materialize(self, frame, variables, period):
        variables = tuple(variables)
        assert variables == UK_TAKE_UP_ENGINE_PREDICTORS, variables
        self.calls.append((variables, str(period)))
        if self.working_age_adult is None:
            values = whole_age_working_age_adult(frame.table("person")["age"])
        else:
            # As given, so a test can hand the stage a malformed answer.
            values = np.asarray(self.working_age_adult)
        return {UK_UC_WORKING_AGE_ADULT: values}
