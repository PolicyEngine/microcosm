"""Regenerate (or check) the committed concept-coverage reports.

Each report compares one engine's concept mapping
(:mod:`microcosm.frame.concept_mapping`) with that engine's input surface:
which inputs each engine-neutral concept feeds, which concepts no input
takes, and which inputs no concept covers. The JSON report is a test golden,
``packages/microcosm-frame/tests/golden/concept-coverage/<engine>.json``,
which each engine's tests compare with the installed engine; the readable
report, ``docs/concept-coverage/<engine>.md``, is rendered from the same data
and checked against the golden by the engine-free tests.

The PolicyEngine reports read the installed engine, so run them in that
engine's environment::

    uv sync --all-packages --locked --extra us
    uv run --no-sync python tools/refresh_concept_coverage.py --engine policyengine-us
    uv sync --all-packages --locked --extra uk
    uv run --no-sync python tools/refresh_concept_coverage.py --engine policyengine-uk

The Axiom reports read the committed, engine-generated input surfaces in
``packages/microcosm-frame/tests/fixtures/axiom_input_surfaces/``
(``tools/refresh_axiom_input_surface.py``) and need no engine::

    uv run --no-sync python tools/refresh_concept_coverage.py --engine axiom-nz

``--check`` exits non-zero, writing nothing, when a committed report
differs from a fresh one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from microcosm.frame.concept_mapping import (
    ConceptMapping,
    CoverageReport,
    InputRef,
    coverage_report,
    rules_engine_input_refs,
)

REPOSITORY = Path(__file__).resolve().parents[1]
REPORTS = REPOSITORY / "docs" / "concept-coverage"
GOLDEN = (
    REPOSITORY
    / "packages"
    / "microcosm-frame"
    / "tests"
    / "golden"
    / "concept-coverage"
)
SURFACES = (
    REPOSITORY
    / "packages"
    / "microcosm-frame"
    / "tests"
    / "fixtures"
    / "axiom_input_surfaces"
)
ENGINES = ("policyengine-us", "policyengine-uk", "axiom-nz", "axiom-be")


def _mapping_and_surface(engine: str) -> tuple[ConceptMapping, tuple[InputRef, ...]]:
    if engine == "policyengine-us":
        from microcosm.frame.adapters.policyengine_us import (
            POLICYENGINE_US_CONCEPT_MAPPING,
            PolicyEngineUSVariableMetadataIndex,
        )

        index = PolicyEngineUSVariableMetadataIndex()
        return POLICYENGINE_US_CONCEPT_MAPPING, rules_engine_input_refs(index)
    if engine == "policyengine-uk":
        from microcosm.frame.adapters.policyengine_uk import (
            POLICYENGINE_UK_CONCEPT_MAPPING,
            PolicyEngineUKEngine,
        )

        return POLICYENGINE_UK_CONCEPT_MAPPING, rules_engine_input_refs(
            PolicyEngineUKEngine()
        )
    from microcosm.frame.adapters.axiom import axiom_concept_mapping
    from microcosm.frame.adapters.axiom_input_surface import load_axiom_input_surface

    country = engine.removeprefix("axiom-")
    surface = load_axiom_input_surface(SURFACES / f"{country}.json")
    return axiom_concept_mapping(country), surface.refs()


def build_report(engine: str) -> CoverageReport:
    """The coverage report for ``engine`` from its mapping and surface."""

    mapping, surface = _mapping_and_surface(engine)
    return coverage_report(mapping, surface)


def _rendered(report: CoverageReport) -> dict[str, str]:
    payload = json.dumps(report.to_dict(), indent=1, sort_keys=True) + "\n"
    return {"json": payload, "md": report.to_markdown()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=ENGINES, required=True, action="append")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale: list[str] = []
    for engine in args.engine:
        report = build_report(engine)
        if report.unknown_inputs:
            labels = ", ".join(ref.label() for ref in report.unknown_inputs)
            print(f"{engine}: mapped inputs missing from the engine: {labels}")
            return 1
        for suffix, text in _rendered(report).items():
            directory = GOLDEN if suffix == "json" else REPORTS
            path = directory / f"{engine}.{suffix}"
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current == text:
                continue
            if args.check:
                stale.append(str(path.relative_to(REPOSITORY)))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(REPOSITORY)}")
    if stale:
        print("stale concept-coverage reports: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
