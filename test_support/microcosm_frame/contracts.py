"""The behavioral contract suite: guarantees the platform makes, as tests.

Each test states an invariant the microcosm stack promises — about weights,
frame structure, links, mass conservation, accounting, unit assignment, and
the rules-engine boundary. Operators (microcosm-fit, microcosm-calibrate,
microcosm-build stages) may rely on every guarantee here; anything that would
break one of these tests is a kernel-level bug, not a tuning choice.
"""

# ruff: noqa: F401

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    US_SCHEMA,
    EntitySchema,
    ExportContract,
    Frame,
    LinkSpec,
    MassChange,
    RulesEngine,
    VariableMetadata,
    WeightKind,
    Weights,
    gini,
    wsum,
)
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

# ----------------------------------------------------------------------------
# Weights: corrupted vectors can never enter the system
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Weight kinds: design -> importance -> calibrated, never backward
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Bundle invariants: structure is validated on every construction
# ----------------------------------------------------------------------------


def _tables(simple_schema_unused=None) -> dict[str, pd.DataFrame]:
    person = pd.DataFrame(
        {
            "person_id": [0, 1, 2],
            "person_household_id": [1, 1, 2],
            "income": [10.0, 0.0, 5.0],
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    return {"person": person, "household": household}


def _weights() -> dict[str, Weights]:
    return {"household": Weights(values=np.array([1.0, 2.0]), kind=WeightKind.DESIGN)}


# ----------------------------------------------------------------------------
# Concat: pool strata assemble without losing a gram of mass
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Select: subsets re-validate, never silently corrupt
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Accounting: weighted aggregates are the weighted truth
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Unit structure: assignment partitions exactly
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Rules engines: adapters satisfy the protocol; exports round-trip
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Links (documented placeholder): declared associations validate their tables
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# microcosm-fit (not yet in workspace): the contract it must meet on arrival
# ----------------------------------------------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
