"""The bonus-sacrifice anchor follows the locked policyengine-uk (microcosm#1340).

The salary_sacrifice stage converts FRS bonuses until the engine's projection
of them meets OBR's 2027-28 base above the GBP 2,000 threshold. These tests
fail if the engine moves the growth it projects either route or the weights
by, the upper earnings limit, or the rule that the threshold applies to the
two routes summed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from microcosm.build.uk_runtime.salary_sacrifice import (
    bonus_sacrifice_excess,
    load_bonus_sacrifice_anchor,
)
from test_support.paths import paths_for

_ROOT = paths_for("microcosm-build").repository


def _tool():
    spec = importlib.util.spec_from_file_location(
        "pin_uk_bonus_sacrifice_anchor",
        _ROOT / "tools/pin_uk_bonus_sacrifice_anchor.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_anchor_is_the_installed_engines() -> None:
    assert _tool().build_anchor() == load_bonus_sacrifice_anchor()


@pytest.mark.parametrize(
    ("salary", "bonus"),
    [(0.0, 1_000.0), (1_500.0, 1_000.0), (3_000.0, 1_000.0), (0.0, 5_000.0)],
)
def test_the_bonus_routes_excess_is_the_engines_cap_arithmetic(
    salary: float, bonus: float
) -> None:
    import policyengine_uk

    def excess(bonus_amount: float) -> float:
        simulation = policyengine_uk.Simulation(
            situation={
                "people": {
                    "worker": {
                        "age": 40,
                        "employment_income": {2029: 100_000},
                        "pension_contributions_via_salary_sacrifice": {2029: salary},
                        "pension_contributions_via_bonus_sacrifice": {
                            2029: bonus_amount
                        },
                    }
                },
                "benunits": {"unit": {"members": ["worker"]}},
                "households": {"home": {"members": ["worker"]}},
            }
        )
        return float(
            np.asarray(
                simulation.calculate("salary_sacrifice_pension_ni_excess", 2029)
            )[0]
        )

    # The threshold is the engine's 2029 cap, which the OBR's base lies above.
    assert excess(bonus) - excess(0.0) == pytest.approx(
        float(
            bonus_sacrifice_excess(
                np.asarray([salary]), np.asarray([bonus]), earnings_growth=1.0
            )[0]
        )
    )
