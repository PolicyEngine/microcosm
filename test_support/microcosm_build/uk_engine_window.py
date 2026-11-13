"""Engine-free stand-ins for the UK measure resolver's engine year window.

A scratch-mode ``UKMeasureResolver`` rewrites its one-year scratch H5 as a
multi-year H5 through its year (policyengine-uk's own uprating) and builds
each simulation inside ``uk_rebalancing_years_through``. Both steps need
policyengine-uk itself, so a test that stands a fake engine in for it keeps
the one-year file and the installed rebalancing loop, and records the
projections the resolver asked for.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from microcosm.build.uk_runtime import measure_simulation


def stub_scratch_engine_window(monkeypatch) -> list[tuple[str, int]]:
    """Stub both window steps; returns the ``(file name, end year)`` requests."""

    projections: list[tuple[str, int]] = []

    def project(path, end_year):
        projections.append((Path(path).name, int(end_year)))

    monkeypatch.setattr(measure_simulation, "_project_scratch_engine_input", project)
    monkeypatch.setattr(
        measure_simulation,
        "uk_rebalancing_years_through",
        lambda end_year: nullcontext(),
    )
    return projections
