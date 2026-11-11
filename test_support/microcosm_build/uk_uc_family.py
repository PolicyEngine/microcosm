"""UC calibration family categories respect retained FRS relationships."""

# ruff: noqa: F401

import json
from importlib import resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.ledger_targets import (
    UKFrameTargetAdapter,
    materialize_uk_ledger_targets,
)
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    compute_uk_measure_input,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _fixture():
    # Single parent + 18-year-old QYP; unmarried parents + child;
    # childless couple including an 18-year-old partner; single claimant;
    # parent + reported 19-year-old who is not eligible for the child element.
    people = pd.DataFrame(
        {
            "person_id": np.arange(10),
            "person_benunit_id": [10, 10, 20, 20, 20, 30, 30, 40, 50, 50],
            "age": [49, 18, 35, 36, 8, 19, 18, 18, 45, 19],
            "is_benunit_head": [1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
            "is_parent": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        }
    )
    benunits = pd.DataFrame(
        {
            "benunit_id": [50, 30, 10, 40, 20],  # deliberately different order
            "dependent_children": [1, 0, 1, 0, 1],
            "is_married": [False] * 5,  # cohabiting couples must remain couples
        }
    )
    frame = SimpleNamespace(
        table=lambda entity: {"person": people, "benunit": benunits}[entity]
    )
    child_variable = "is_child_or_qualifying_young_person_for_universal_credit"
    sim = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables={}),
        calculate=lambda variable, year: (
            np.array([False, True, False, False, True, True, True, True, False, False])
            if variable == child_variable
            else None
        ),
    )
    return frame, sim


def _administrative_fixture(allowances=None):
    frame, original = _fixture()
    # In benefit-unit order: lone parent; structural young couple receiving a
    # single allowance; lone parent; unavailable allowance; cohabiting parents.
    # The single-allowance couple is an explicit input scenario, not an assertion
    # that the engine reconstructs administrative partner eligibility.
    values = np.array([1200, 1200, 1200, 0, 1800], dtype=np.float32)
    if allowances is not None:
        values = np.asarray(allowances)

    def parameters(period):
        assert period == "2025"
        return SimpleNamespace(
            gov=SimpleNamespace(
                dwp=SimpleNamespace(
                    universal_credit=SimpleNamespace(
                        standard_allowance=SimpleNamespace(
                            amount=SimpleNamespace(SINGLE_YOUNG=80, SINGLE_OLD=100)
                        )
                    )
                )
            )
        )

    def calculate(variable, year):
        if variable == "uc_standard_allowance":
            return values
        if variable == "uc_child_element":
            # A reported 19-year-old need not attract a child element; a young
            # claimant qualifying as a QYP does not make their own claim entitled.
            return np.array([0, 0, 100, 0, 100])
        if variable == "universal_credit":
            return np.array([100, 100, 0, 0, 100])
        return original.calculate(variable, year)

    return frame, SimpleNamespace(
        calculate=calculate,
        tax_benefit_system=SimpleNamespace(parameters=parameters, variables={}),
    )


__all__ = [name for name in globals() if not name.startswith("__")]
