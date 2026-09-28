# ruff: noqa: F401
from __future__ import annotations

import json
from dataclasses import replace
from importlib import metadata, resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import (
    SourceManifest,
    SourceOperationSpec,
    SourceStageSpec,
)
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_brma import (
    UKFRSBRMAStageTransform,
    assign_brma_by_cell,
    collapse_benunit_brma_to_household,
)
from microcosm.build.uk_runtime.frs_household_draws import (
    FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS,
    UKFRSHouseholdDrawsStageTransform,
)
from microcosm.build.uk_runtime.frs_person_draws import (
    FRS_PERSON_DRAW_OUTPUT_COLUMNS,
    UKFRSPersonDrawsStageTransform,
    derive_frs_person_draws,
)
from microcosm.build.uk_runtime.frs_take_up import (
    FRS_TAKE_UP_OUTPUT_COLUMNS,
    UKFRSTakeUpStageTransform,
    UKTakeUpPopulationPolicy,
    aggregate_person_reported_to_benunit,
    assert_take_up_stage_population_declaration,
    derive_frs_take_up,
    uc_age_eligible_benunits,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame

# The engine's 2025 working-age bounds (is_adult at 18, State Pension age 66),
# injected so the hermetic tests need no engine; the lockstep test below checks
# the reader returns exactly this.
_POLICY = UKTakeUpPopulationPolicy(
    adult_age=18, state_pension_age=66, instant="2025-01-01", source="test"
)


def _take_up_stage() -> SourceStageSpec:
    manifest = SourceManifest.from_mapping(
        json.loads(
            resources.files("microcosm.build.uk")
            .joinpath("source_stages.json")
            .read_text(encoding="utf-8")
        )
    )
    return next(stage for stage in manifest.stages if stage.stage == "frs_take_up")


class _Contract:
    rates = {
        "child_benefit": 0.5,
        "child_benefit_opts_out_rate": 0.0,
        "pension_credit": 0.5,
        "universal_credit": 0.5,
        "tax_free_childcare": 0.5,
        "extended_childcare": 0.5,
        "universal_childcare": 0.5,
        "targeted_childcare": 0.5,
        "uc_childcare_single": 0.5,
        "uc_childcare_couple": 0.0,
        "marriage_allowance": 0.5,
        "scp_under_6": 0.97,
        "scp_6_plus": 0.85,
        "tv_ownership_rate": 0.5,
        "tv_licence_evasion_rate": 0.5,
        "first_time_buyer_rate": 0.5,
        "property_purchase_rate": 0.5,
        "tax_free_childcare_spend_routed_share": 0.593,
    }

    def rate(self, key: str, build_year: int | None = None) -> float:
        return self.rates[key]

    def continuous_entry(self, key: str):
        assert key == "maximum_extended_childcare_hours_usage"
        return {"mean": 15.019, "sd": 4.972, "lower": 0, "upper": 30}

    def entry(self, key: str):
        assert key == "tax_free_childcare_spend_routed_share"
        return SimpleNamespace(raw={"entity": "person"})

    build_year = 2024


class _FakeEngine:
    country = "uk"

    def __init__(self, lha_category):
        self.lha_category = lha_category

    def materialize(self, frame, variables, period):
        assert tuple(variables) == ("LHA_category",)
        assert period == "2024"
        return {"LHA_category": self.lha_category}


def _frame() -> object:
    person = pd.DataFrame(
        {
            "person_id": [101, 102, 201, 301],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [1, 1, 1, 2],
            "age": [5, 6, 40, 70],
            "child_benefit_reported": [0, 10, 0, 0],
            "pension_credit_reported": [0, 0, 0, 5],
            "universal_credit_reported": [0, 0, 20, 0],
        }
    )
    benunit = pd.DataFrame(
        {"benunit_id": [10, 20, 30], "is_married": [False, True, False]}
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "SCOTLAND"],
            "household_weight": [2.0, 3.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
    )


__all__ = [name for name in globals() if not name.startswith("__")]
