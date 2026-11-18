"""The population contract's stand-in index answers as the installed engine's.

The contract's engine-free tests inject ``StandInVariableIndex.from_lock()``
in place of the default :class:`PolicyEngineUSVariableIndex`, which reads the
installed policyengine-us sources. These tests pin the stand-in to that
default: over every name the stand-in classifies, the two give the same
known, computed-at-2024 and input-at-2024 answers, so a passing engine-free
clause test means what it would mean with the engine's metadata behind it.
"""

from __future__ import annotations

import pandas as pd

from microcosm.build.us_runtime.multispine_pool import (
    POOL_ENGINE_INPUT_PROJECTION_CONTRACT,
)
from microcosm.build.us_runtime.one_population_contract import (
    STRUCTURAL_COLUMNS,
    PolicyEngineUSVariableIndex,
)
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)
from test_support.microcosm_build.us_one_population_contract import (
    SPM_ROLE,
    STAND_IN_COMPAT_AGGREGATES,
    STAND_IN_COMPUTED_AT_2024,
    STAND_IN_FORMULA_AFTER_2024,
    STAND_IN_STRUCTURAL_FORMULAS,
    StandInVariableIndex,
    lock_engine_inputs,
)


def test_the_lock_inputs_are_the_installed_engine_inputs() -> None:
    # The stand-in's inputs are the lock's 926 names; the installed index's
    # year-blind input list must be the same names.
    installed = frozenset(PolicyEngineUSVariableMetadataIndex().variables())
    assert installed == frozenset(lock_engine_inputs())
    assert len(installed) == POOL_ENGINE_INPUT_PROJECTION_CONTRACT.input_count


def _every_classified_name() -> dict[str, pd.DataFrame]:
    names = sorted(
        frozenset(lock_engine_inputs())
        | STAND_IN_COMPUTED_AT_2024
        | STAND_IN_COMPAT_AGGREGATES
        | STAND_IN_FORMULA_AFTER_2024
        | STAND_IN_STRUCTURAL_FORMULAS
        | STRUCTURAL_COLUMNS
    )
    return {"person": pd.DataFrame(columns=names)}


def test_the_stand_in_answers_as_the_default_index() -> None:
    tables = _every_classified_name()
    names = list(tables["person"].columns)
    default = PolicyEngineUSVariableIndex()
    stand_in = StandInVariableIndex.from_lock()

    assert default.known_variables(names) == stand_in.known_variables(names)
    assert default.computed_at_2024(tables) == stand_in.computed_at_2024(tables)
    assert default.inputs_at_2024(tables) == stand_in.inputs_at_2024(tables)


def test_a_post_2024_formula_and_the_role_are_inputs_at_2024() -> None:
    # weeks_worked has only formula_2025 at the locked policyengine-us, and
    # the SPM role is a declared dataset source input despite its fallback
    # formula: the contract reads both as inputs at 2024.
    tables = {"person": pd.DataFrame(columns=["weeks_worked", SPM_ROLE])}
    inputs = PolicyEngineUSVariableIndex().inputs_at_2024(tables)
    assert inputs == {"weeks_worked", SPM_ROLE}
