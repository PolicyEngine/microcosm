"""US validation-input coverage gate: no validation row scores a structural zero.

Replays the invisible-gap class the gate exists to catch (microcosm#252/#253):
a validation config scores a provision whose effect is driven by a pure-input
leaf that no source stage produces, so the row validates as a silent zero. The
plant-a-missing-leaf test proves the gate fails loudly in exactly that case; the
registry-consistency test proves the seed entries are real provision input
leaves against the live PolicyEngine-US graph (so the registry cannot rot).
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import pytest

from microcosm.build.us_runtime import (
    US_QBI_OUTPUT_COLUMNS,
    US_VALIDATION_PROVISION_INPUT_LEAVES,
    ValidationInputLeaf,
    assert_validation_leaf_registry_current,
    us_source_stage_outputs,
    us_validation_input_coverage_gate,
)
from microcosm.build.us_runtime.validation_input_coverage import (
    us_validation_input_leaf_requirements,
    us_validation_input_leaf_reviewed_exclusions,
)

__all__ = [name for name in globals() if not name.startswith("__")]
