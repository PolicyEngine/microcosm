# ruff: noqa: F401
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.build.uk_runtime.uc_deduction_attributes import (
    FLOAT32_UNIFORM_MAX,
    UC_DEDUCTION_BANDS,
    UC_DEDUCTION_COMBINATIONS,
    UC_DEDUCTION_OUTPUT_COLUMNS,
    UC_DEDUCTION_REGIONS,
    UC_DEDUCTION_RESOURCE,
    UKUCDeductionAttributesStageTransform,
    _identity_float32_uniforms,
    load_uc_deduction_distributions,
    map_uniform_to_banded_rate,
    map_uniform_to_categorical,
    validate_uc_deduction_resource,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository
RESOURCE = (
    ROOT / "packages/microcosm-build/src/microcosm/build/uk" / UC_DEDUCTION_RESOURCE
)


def _stage_mapping() -> dict[str, object]:
    return {
        "stage": "uc_deduction_attributes",
        "survey": "test",
        "source": "test",
        "grain": "benunit",
        "artifacts": [],
        "operations": [
            {
                "kind": "assign_uniform_draw",
                "output": "uc_deduction_random_draw",
                "seed": 0,
            },
            {
                "kind": "assign_uniform_draw",
                "output": "uc_deduction_type_random_draw",
                "seed": 0,
            },
            {
                "kind": "map_uniform_to_banded_rate",
                "output": "uc_latent_deduction_rate",
                "draw": "uc_deduction_random_draw",
                "draw_dtype": "float32",
                "resource": UC_DEDUCTION_RESOURCE,
                "distribution": "latent_rate_distribution",
                "incidence_modifier": {
                    "column": "region",
                    "entity": "household",
                    "table": "region_incidence_factor",
                },
                "none_value": 0.0,
            },
            {
                "kind": "map_uniform_to_categorical",
                "output": "uc_deduction_combination",
                "draw": "uc_deduction_type_random_draw",
                "draw_dtype": "float32",
                "resource": UC_DEDUCTION_RESOURCE,
                "distribution": "type_combination",
                "gate": {
                    "column": "uc_latent_deduction_rate",
                    "positive": True,
                },
                "none_value": "NONE",
            },
        ],
        "outputs": list(UC_DEDUCTION_OUTPUT_COLUMNS),
        "nonnegative_outputs": list(UC_DEDUCTION_OUTPUT_COLUMNS[:3]),
    }


def _stage() -> SourceStageSpec:
    return SourceStageSpec.from_mapping(_stage_mapping())


def _frame(n: int = 512, *, region_names: tuple[str, ...] | None = None):
    ids = np.arange(1, n + 1, dtype=np.int64)
    regions = np.resize(
        np.asarray(region_names or tuple(sorted(UC_DEDUCTION_REGIONS)), dtype=object),
        n,
    )
    person = pd.DataFrame(
        {
            "person_id": ids * 10 + 1,
            "person_benunit_id": ids,
            "person_household_id": ids,
            "age": np.full(n, 40),
            "is_benunit_head": np.ones(n, dtype=bool),
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": ids,
            "would_claim_uc": np.ones(n, dtype=bool),
            "universal_credit_pre_benefit_cap": np.full(n, 500.0),
            "benefit_cap_reduction": np.zeros(n),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": regions,
            "council_tax": np.zeros(n),
            "tenure_type": np.full(n, "OWNED_OUTRIGHT", dtype=object),
            "rent": np.zeros(n),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=np.ones(n),
        time_period="2024",
    )


def _engine_splitmix64_uniform(ids: np.ndarray, salt: int = 0) -> np.ndarray:
    """PolicyEngine-UK's fallback draw (``utils/stochastic.py`` at 2.92.1), inlined."""

    with np.errstate(over="ignore"):
        z = ids.astype(np.uint64) + np.uint64(salt) * np.uint64(0x632BE59BD9B4E019)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    draws = (z >> np.uint64(11)).astype(np.float64) / 2.0**53
    return np.minimum(draws, 1.0 - 2.0**-24)


__all__ = [name for name in globals() if not name.startswith("__")]
