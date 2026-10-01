"""Every registered concept mapping, and the committed evidence beside them.

Shared by the engine-free contract tests, the country-specific mapping tests
and the engine tests, so each reads the same mappings, input surfaces and
coverage goldens from one place.
"""

# ruff: noqa: F401

from __future__ import annotations

from pathlib import Path

from microcosm.frame.adapters.axiom import axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import (
    AxiomInputSurface,
    load_axiom_input_surface,
)
from microcosm.frame.adapters.policyengine_uk import POLICYENGINE_UK_CONCEPT_MAPPING
from microcosm.frame.adapters.policyengine_us import POLICYENGINE_US_CONCEPT_MAPPING
from microcosm.frame.concept_mapping import ConceptMapping
from test_support.paths import paths_for

_FRAME = paths_for("microcosm-frame")
AXIOM_COUNTRIES = ("nz", "be")
AXIOM_SURFACES = _FRAME.tests / "fixtures" / "axiom_input_surfaces"
COVERAGE_GOLDEN = _FRAME.tests / "golden" / "concept-coverage"
COVERAGE_DOCS = _FRAME.repository / "docs" / "concept-coverage"


def concept_mappings() -> dict[str, ConceptMapping]:
    """Every committed mapping, keyed by its coverage-report name."""

    mappings = {
        "policyengine-us": POLICYENGINE_US_CONCEPT_MAPPING,
        "policyengine-uk": POLICYENGINE_UK_CONCEPT_MAPPING,
    }
    for country in AXIOM_COUNTRIES:
        mappings[f"axiom-{country}"] = axiom_concept_mapping(country)
    return mappings


def axiom_input_surface(country: str) -> AxiomInputSurface:
    """The committed, engine-generated input surface for ``country``."""

    surface = load_axiom_input_surface(AXIOM_SURFACES / f"{country}.json")
    assert surface.country == country
    return surface


def coverage_golden(name: str) -> Path:
    """The committed coverage-report golden for ``name``."""

    return COVERAGE_GOLDEN / f"{name}.json"


__all__ = [name for name in globals() if not name.startswith("__")]
