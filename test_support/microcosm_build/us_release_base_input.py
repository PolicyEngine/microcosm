"""A synthetic base frame and the release's re-runs of its QRF stages.

Shared by the release base-input invariant tests. The base is built with the
production stage functions in ``tools/build_us_puf_support_base.py`` order
(childcare, energy subsidy, retirement contributions, once before the PUF
support clone and once after), and the release frames apply the release
builder's own repair functions before re-running those stages as
``tools/build_us_fiscal_refresh_release.py`` does when it has no validated
pool frame.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.childcare import (
    US_CHILDCARE_OUTPUT_COLUMNS,
    with_us_childcare_inputs,
)
from microcosm.build.us_runtime.energy_subsidy import (
    US_ENERGY_SUBSIDY_OUTPUT_COLUMNS,
    with_us_energy_subsidy_input,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    with_us_retirement_contribution_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

TEST_PATHS = paths_for("microcosm-build")
RELEASE_TOOL = TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
BASE_TOOL = TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"
SEED = 0
PERIOD = 2024
N_PEOPLE = 320

# The Route A release's recorded Social Security component factors
# (calibration_diagnostics.json build.support_value_repairs, release
# populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z).
ROUTE_A_SS_FACTORS = {
    "social_security_retirement": 1.1474528249656013,
    "social_security_disability": 1.1103656007127674,
    "social_security_survivors": 2.4926129688735283,
    "social_security_dependents": 0.9056065944636575,
}


def childcare(frame: Frame) -> Frame:
    return with_us_childcare_inputs(
        frame, seed=SEED, time_period=PERIOD, allow_existing_without_source=True
    )


def energy_subsidy(frame: Frame) -> Frame:
    return with_us_energy_subsidy_input(
        frame, seed=SEED, time_period=PERIOD, allow_existing_without_source=True
    )


def retirement_contributions(frame: Frame) -> Frame:
    return with_us_retirement_contribution_inputs(frame, seed=SEED, time_period=PERIOD)


# Stage -> (entity, output columns, the call as the release makes it). The base
# builder makes the same calls without ``allow_existing_without_source``, which
# only matters when the raw ASEC source columns are absent.
STAGES: dict[str, tuple[str, tuple[str, ...], Callable[[Frame], Frame]]] = {
    "childcare": ("spm_unit", US_CHILDCARE_OUTPUT_COLUMNS, childcare),
    "energy_subsidy": ("spm_unit", US_ENERGY_SUBSIDY_OUTPUT_COLUMNS, energy_subsidy),
    "retirement_contributions": (
        "person",
        US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
        retirement_contributions,
    ),
}


def load_release_tool():
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_invariant", RELEASE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def asec_frame(n: int = N_PEOPLE, seed: int = 11) -> Frame:
    """One person per unit, carrying every raw source the three stages read."""

    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype=np.int64)
    age = rng.integers(18, 86, n)
    wages = np.where(rng.random(n) < 0.6, np.round(rng.lognormal(10.6, 0.8, n)), 0.0)
    self_employment = np.where(
        rng.random(n) < 0.15, np.round(rng.lognormal(9.8, 1.0, n)), 0.0
    )

    def benefit(mask: np.ndarray, mean: float, sd: float) -> np.ndarray:
        return np.where(mask, np.maximum(np.round(rng.normal(mean, sd, n)), 1.0), 0.0)

    earned = (wages > 0) | (self_employment > 0)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids * 10,
            "person_tax_unit_id": ids * 100,
            "person_spm_unit_id": ids * 1_000,
            "person_family_id": ids * 10_000,
            "person_marital_unit_id": ids * 100_000,
            "age": age,
            "is_female": rng.random(n) < 0.5,
            "has_esi": (wages > 0) & (rng.random(n) < 0.7),
            "tax_unit_role_input": ["HEAD"] * n,
            "WSAL_VAL": wages,
            "SEMP_VAL": self_employment,
            "employment_income_before_lsr": wages,
            "self_employment_income_before_lsr": self_employment,
            "RETCB_VAL": np.where(
                earned & (rng.random(n) < 0.5),
                np.round((wages + self_employment) * rng.uniform(0.01, 0.12, n)),
                0.0,
            ),
            "SPM_CHILDCAREXPNS": np.where(
                (age < 50) & (rng.random(n) < 0.25),
                np.round(rng.lognormal(8.5, 0.7, n)),
                0.0,
            ),
            "SPM_ENGVAL": np.where(
                rng.random(n) < 0.12, np.round(rng.lognormal(6.0, 0.5, n)), 0.0
            ),
            "social_security_retirement": benefit(
                (age >= 62) & (rng.random(n) < 0.75), 21_000, 6_000
            ),
            "social_security_disability": benefit(
                (age < 62) & (rng.random(n) < 0.1), 15_000, 4_000
            ),
            "social_security_survivors": benefit(rng.random(n) < 0.08, 14_000, 5_000),
            "social_security_dependents": benefit(rng.random(n) < 0.06, 8_000, 3_000),
        }
    )
    tables = {
        entity: pd.DataFrame({f"{entity}_id": ids * scale})
        for entity, scale in (
            ("household", 10),
            ("tax_unit", 100),
            ("spm_unit", 1_000),
            ("family", 10_000),
            ("marital_unit", 100_000),
        )
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = np.where(
        rng.random(n) < 0.45, "JOINT", "SINGLE"
    )
    weights = rng.uniform(500.0, 3_000.0, n)
    return Frame(tables, US_SCHEMA, {"household": Weights(weights, WeightKind.DESIGN)})


def replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def with_scaled_weights(frame: Frame, factor: float) -> Frame:
    weights = frame.weights_for("household")
    return Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {"household": Weights(weights.values * factor, weights.kind)},
        frame.strata,
    )


def build_base(n: int = N_PEOPLE, seed: int = 11) -> Frame:
    """ASEC pass, PUF-support clone, PUF-half income, then the post-clone pass."""

    frame = asec_frame(n, seed)
    for _, _, call in STAGES.values():
        frame = call(frame)
    expanded = clone_us_frame_for_puf_support(frame)
    person = expanded.table("person").copy()
    puf = person["person_support_clone_index"].astype(int).to_numpy() > 0
    rng = np.random.default_rng(seed + 18)
    # Stand-in for the PUF tax-detail QRF: the clone half carries different
    # wages, self-employment income and Social Security from its ASEC twin.
    for column in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        *ROUTE_A_SS_FACTORS,
    ):
        values = person[column].to_numpy(dtype=np.float64).copy()
        values[puf] = np.round(values[puf] * rng.lognormal(0.0, 0.35, puf.sum()))
        person[column] = values
    frame = replace_person(expanded, person)
    for _, _, call in STAGES.values():
        frame = call(frame)
    return frame


def outputs(frame: Frame, stage: str) -> pd.DataFrame:
    entity, columns, _ = STAGES[stage]
    return frame.table(entity).set_index(f"{entity}_id")[list(columns)].sort_index()


def ss_targets(release_tool, frame: Frame) -> list[SimpleNamespace]:
    """SSA targets that make the release repair apply the Route A factors."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    role_by_column = {
        column: role
        for role, column in release_tool.US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.items()
    }
    return [
        SimpleNamespace(
            metadata={"target_role": role_by_column[column]},
            value=float(np.sum(person[column].to_numpy(dtype=np.float64) * weights))
            * factor,
        )
        for column, factor in ROUTE_A_SS_FACTORS.items()
    ]


class ReleaseScenario:
    """The synthetic base, the release's repaired frames, memoized re-runs."""

    def __init__(self, release_tool) -> None:
        self.release_tool = release_tool
        self.base = build_base()
        self.mass_repaired, mass = release_tool._with_base_population_mass_repair(
            self.base
        )
        assert mass["applied"] and mass["factor"] != 1.0
        self.release_prefix, ss = (
            release_tool._with_social_security_component_value_repair(
                self.mass_repaired, ss_targets(release_tool, self.mass_repaired)
            )
        )
        assert ss["applied"]
        self._reruns: dict[tuple[str, str], pd.DataFrame] = {}

    def rerun(self, start: str, stage: str) -> pd.DataFrame:
        """Outputs of ``stage`` re-run on ``base``, ``mass_repaired`` or
        ``release_prefix`` (mass repair, then Social Security repair)."""

        key = (start, stage)
        if key not in self._reruns:
            frame = {
                "base": self.base,
                "mass_repaired": self.mass_repaired,
                "release_prefix": self.release_prefix,
            }[start]
            self._reruns[key] = outputs(STAGES[stage][2](frame), stage)
        return self._reruns[key]
