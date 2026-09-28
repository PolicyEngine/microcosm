"""UK enhanced-FRS release input-column coverage, isolated from PE-UK."""

# ruff: noqa: F401

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS,
    UK_LOADER_INPUT_ALIASES,
    PolicyEngineUKCoverageEngine,
    UKEffectiveMassCoveragePolicy,
    UKReleaseInputColumn,
    UKReleaseInputCoverageManifest,
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
    load_efrs_parity_known_gaps,
    load_efrs_parity_reference,
    load_uk_release_input_coverage_manifest,
    uk_release_input_coverage_gate,
)
from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_REPO_ROOT = _TEST_PATHS.repository
_SHIPPED_MANIFEST = (
    _REPO_ROOT
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "uk"
    / "release_input_coverage_manifest.json"
)


def _person_frame(columns: dict[str, np.ndarray]) -> Frame:
    n = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_benunit_id": np.ones(n, dtype="int64"),
            "person_household_id": np.ones(n, dtype="int64"),
            **{name: np.asarray(values) for name, values in columns.items()},
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.asarray([1], dtype="int64")})
    household = pd.DataFrame({"household_id": np.asarray([1], dtype="int64")})
    return Frame(
        {"person": person, "benunit": benunit, "household": household},
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(values=np.asarray([1000.0]), kind=WeightKind.DESIGN)},
    )


def _weighted_person_frame(
    columns: dict[str, np.ndarray],
    household_weights: np.ndarray,
    *,
    weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
    time_period: str | None = None,
) -> Frame:
    weights = np.asarray(household_weights, dtype=float)
    n = len(weights)
    ids = np.arange(1, n + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            **{name: np.asarray(values) for name, values in columns.items()},
        }
    )
    return Frame(
        {
            "person": person,
            "benunit": pd.DataFrame({"benunit_id": ids}),
            "household": pd.DataFrame({"household_id": ids}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(values=weights, kind=weight_kind)},
        mass_log=mass_log,
        metadata=None if time_period is None else {"time_period": time_period},
    )


class _StubEngine:
    def __init__(
        self,
        defaults: dict[str, object],
        variables: set[str] | None = None,
        entities: dict[str, str] | None = None,
    ) -> None:
        self._defaults = dict(defaults)
        self._variables = set(variables or defaults)
        reference_entities = dict(load_efrs_parity_reference().input_entities)
        self._entities = {
            name: reference_entities.get(name, "person") for name in self._variables
        }
        self._entities.update(entities or {})

    def default_values(self, names) -> dict[str, object]:
        return {name: self._defaults[name] for name in names if name in self._defaults}

    def variables(self) -> list[str]:
        return sorted(self._variables)

    def variable_entities(self, names) -> dict[str, str]:
        return {name: self._entities[name] for name in names if name in self._entities}


def _manifest(
    columns: tuple[UKReleaseInputColumn, ...],
    *,
    family_coverage: dict[str, dict[str, object]] | None = None,
) -> UKReleaseInputCoverageManifest:
    return UKReleaseInputCoverageManifest(
        reference={"source": "test"},
        candidate_evidence={"source": "test"},
        columns=columns,
        family_coverage=family_coverage or {},
    )


_CONTRACT = _manifest(
    (
        UKReleaseInputColumn("employment_income", "required"),
        UKReleaseInputColumn("dividend_income", "required"),
        UKReleaseInputColumn(
            "property_income",
            "reviewed_exclusion",
            reason="not yet ported from enhanced FRS pipeline — pending review",
            tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
        ),
    )
)
_DEFAULTS = {
    "employment_income": 0.0,
    "dividend_income": 0.0,
    "property_income": 0.0,
}


def _hmrc_family_coverage() -> dict[str, dict[str, object]]:
    return {
        "hmrc_spi_income": {
            "status": "required_at_build",
            "stage": "hmrc_spi_income",
            "effective_mass_requirements": {
                "gift_aid": {
                    "status": "distributional_required",
                    "minimum_nondefault_mass_share": 1e-6,
                    "support_channel_column": "person_support_channel",
                    "required_support_channel": "spi",
                    "mass_share_denominator": "all_person_effective_mass",
                }
            },
        }
    }


def _reviewed_gift_aid_exclusion() -> UKReleaseInputColumn:
    return UKReleaseInputColumn(
        "gift_aid",
        "reviewed_exclusion",
        reason="not yet ported from enhanced FRS pipeline — pending review",
        tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
    )


__all__ = [name for name in globals() if not name.startswith("__")]
