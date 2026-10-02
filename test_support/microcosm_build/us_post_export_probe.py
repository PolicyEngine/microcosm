"""Fixtures for the post-export probe's end-to-end tests.

``tools/probe_us_post_export.py`` runs the release tool's post-export stages on
a subsample written by ``tools/sample_us_export_households.py``. These helpers
run the whole chain on the sampler fixtures' nested synthetic export, scoring
it with the post-export scoring tests' fake engine
(:mod:`test_support.microcosm_build.us_post_export_scoring`) made to behave as
policyengine-core does where the probe depends on it: weights come back as
float32 (``household_weight`` is a float variable, and core stores floats as
float32) and are uprated by a population ratio at a later period
(``household_weight``'s ``uprating`` parameter). A fake returning the frame's
exact float64 weights hid the probe's original exact-weight decomposition,
which refused every probe on a real engine.

Unlike :mod:`test_support.microcosm_build.us_export_subsample`, this module
imports the release tool and ``microcosm.build.us_runtime``; with policyengine
-us installed that import builds the engine's tax-benefit system once.
"""

# ruff: noqa: F401, F403, F405

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.release_input_coverage import ReformCoverageProbe
from microcosm.data.stored_inputs import CertifiedEngine
from test_support.microcosm_build.us_export_subsample import *
from test_support.microcosm_build.us_post_export_scoring import (
    _EngineLog,
    _fake_engine,
    _FakeSeries,
    _native_entity,
    builder,
    fake_reforms,
)

#: Population ratio the fake engine applies to 2026 weights (the published
#: 2026 / 2024 census population ratio is 352,304,000 / 346,588,000).
FAKE_UPRATING = {2026: 352_304_000 / 346_588_000}


def float32_engine(log: _EngineLog, *, uprating=FAKE_UPRATING):
    """The scoring tests' fake engine with policyengine-core's weight dtype
    (float32) and its uprating of weights to a later period."""
    base = _fake_engine(log)

    class Float32Microsimulation(base):
        def calculate(self, variable, period, map_to=None):
            series = super().calculate(variable, period, map_to=map_to)
            factor = np.float32(uprating.get(int(period), 1.0))
            weights = series.weights.astype(np.float32) * factor
            return _FakeSeries(np.asarray(series), weights.astype(np.float64))

    return Float32Microsimulation


def engine_probe(
    probe_id: str,
    binding_inputs,
    *,
    measure: str = "income_tax",
    period: int = 2024,
    floor: float = 1.0,
) -> ReformCoverageProbe:
    """A neutralization probe over the synthetic export's inputs."""
    return ReformCoverageProbe(
        id=probe_id,
        name=probe_id,
        parameter_changes={},
        neutralized_variable=binding_inputs[0],
        budget_measure=measure,
        period=period,
        binding_inputs=tuple(binding_inputs),
        min_abs_effect=float(floor),
        effect_direction="baseline_minus_reform",
        expected_sign="either",
        reason="fixture",
        issue="PolicyEngine/microcosm#956",
    )


def fixture_engine_probes() -> tuple[ReformCoverageProbe, ...]:
    """A take-all rare probe, a common one, and a 2026 SPM-unit probe (its
    engine weights are uprated)."""
    return (
        engine_probe("rare_keogh", ("keogh_distributions",)),
        engine_probe("common_wages", ("employment_income_before_lsr",)),
        engine_probe(
            "snap_take_up_2026",
            ("takes_up_snap_if_eligible",),
            measure="snap",
            period=2026,
        ),
    )


def fixture_engine(frame, *, missing: tuple[str, ...] = ()) -> CertifiedEngine:
    """An "installed engine" defining every stored column except ``missing``
    (``household_weight`` included: the writer stores it from the weights)."""
    columns = {
        column
        for entity in US_ENTITIES
        for column in (*frame.table(entity).columns, "household_weight")
        if column not in missing
    }
    return CertifiedEngine(label="fixture engine", variables=frozenset(columns))


def run_fixture_probe(
    probe_tool,
    release_tool,
    export_path: Path,
    out_dir: Path,
    *,
    receipt=None,
    probes=None,
    batch_size: int | None = None,
    fail_on: str | None = None,
    **options,
):
    """Run the probe with the float32 fake engine; return (report, log).

    ``fail_on`` names a variable the fake engine refuses to compute.
    """
    log = _EngineLog()
    log.fail_on = fail_on
    report = probe_tool.probe_export(
        export_path,
        out_dir,
        sample_receipt=receipt,
        probes=fixture_engine_probes() if probes is None else probes,
        builder=release_tool,
        sampler=_load_tool(
            "sample_us_export_households", "sample_us_export_households.py"
        ),
        batch_size=batch_size,
        scorer_options={
            "microsimulation_cls": float32_engine(log),
            "dataset_from_frame": lambda batch_frame: batch_frame,
        },
        load_frame=load_table_h5,
        measure_entity=_native_entity,
        chunk_rows=7,
        **options,
    )
    return report, log


def verdicts_by_check(report) -> dict[tuple[str, str], dict]:
    return {(row["stage"], row["check"]): row for row in report["verdicts"]}


__all__ = [name for name in globals() if not name.startswith("__")]
