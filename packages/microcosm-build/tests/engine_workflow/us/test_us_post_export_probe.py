"""The export subsampler and post-export probe against policyengine-us.

An eight-household release H5 written through the engine's own writer is
subsampled with the sampler's defaults (the release writer and the deny-list
boundary), and the probe scores the subsample in at least three batches. Two
neutralization probes, one at 2024 and one at 2026, must decompose: the
per-household effects summed with the engine's own row weights reproduce the
gate's effects, and the row-to-household mapping recovers the engine's float32
weights at one factor per key, 1 at the data's period and the population
uprating ratio at 2026.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import math
from pathlib import Path

import pytest

from test_support.microcosm_build.us_post_export_probe import *
from test_support.microcosm_build.us_post_export_scoring import (
    _GUARD_HOUSEHOLDS,
    _engine_probe,
    _write_engine_h5,
)


@pytest.mark.slow
def test_probe_decomposes_policyengine_us_effects(
    builder, sampler, probe_tool, tmp_path
) -> None:
    source = _write_engine_h5(builder, tmp_path / "export", _GUARD_HOUSEHOLDS)
    probes = (
        _engine_probe(
            "wages_income_tax_2024",
            "income_tax",
            2024,
            neutralized_variable="employment_income_before_lsr",
        ),
        _engine_probe(
            "wages_state_income_tax_2026",
            "state_income_tax",
            2026,
            neutralized_variable="employment_income_before_lsr",
        ),
    )
    receipt = sampler.sample_export(
        source,
        tmp_path / "sample",
        fraction=0.5,
        seed=0,
        certainty_threshold=0.0,
        probes=probes,
    )
    assert receipt["verification"]["passed"]
    assert receipt["selection"]["households"] == 4
    report = probe_tool.probe_export(
        Path(receipt["output"]["path"]),
        tmp_path / "probe",
        sample_receipt=receipt,
        probes=probes,
        builder=builder,
        sampler=sampler,
        stages=("stored_inputs", "reform_coverage_smoke", "demographics"),
    )
    errors = {
        name: record.get("traceback")
        for name, record in report["stages"].items()
        if record["status"] != "completed"
    }
    assert not errors, errors
    assert report["stages"]["open_scorer"]["n_batches"] >= probe_tool.MINIMUM_BATCHES
    rows = {
        row["probe"]: row for row in report["stages"]["reform_coverage_smoke"]["probes"]
    }
    for row in rows.values():
        assert row["decomposition"] == "decomposed", row
        assert row["effect"] != 0.0, row
        assert row["standard_error"] is not None
        assert math.isfinite(row["standard_error"])
    assert rows["wages_income_tax_2024"]["engine_weight_scale"] == pytest.approx(
        1.0, rel=1e-6
    )
    # policyengine-us uprates household_weight by the census population ratio
    # (household_weight's uprating parameter), so 2026 rows weigh more.
    from policyengine_us import Microsimulation

    population = Microsimulation.default_tax_benefit_system_instance.parameters.calibration.gov.census.populations.total
    uprating = population("2026-01-01") / population("2024-01-01")
    assert uprating > 1.0
    assert rows["wages_state_income_tax_2026"]["engine_weight_scale"] == pytest.approx(
        uprating, rel=1e-5
    )
    assert rows["wages_income_tax_2024"]["measure_entity"] == "tax_unit"
