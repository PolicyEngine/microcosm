"""Disclosure control in the UK weighted-integrity measurement recorder.

The #609 measurement pass exists to be posted on #578, so its output is a
publication under UK Data Service End User Licence CD137 v16.00 clause 8,
which binds it to the statistical disclosure control standards in
CD171-ResearchDataHandling §5.2.1: no unit-record values (maxima and minima
named explicitly), and nothing reported from one or two cases.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _measurement_module():
    path = ROOT / "tools" / "measure_uk_weighted_integrity_baselines.py"
    spec = importlib.util.spec_from_file_location(
        "measure_uk_weighted_integrity_baselines",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MEASUREMENT = _measurement_module()


def test_no_unit_record_value_is_ever_emitted() -> None:
    """CD171 §5.2.1(3): maxima and minima must be avoided in outputs."""

    values = np.array([1.0, 2.0, 3.0, 4_000_000.0] * 10)
    weights = np.ones(values.size)

    measured = MEASUREMENT._tail_measurements(values, weights, (10,))

    assert "max_abs_value" not in measured
    assert not any("max" in key or "min" in key for key in measured)
    # The population aggregate is still reported: it sums every carrier.
    assert measured["total_weighted_abs_mass"] == float(np.abs(values).sum())


def test_thin_column_suppresses_concentration_and_carrier_count() -> None:
    values = np.array([5.0, 7.0, 0.0, 0.0])
    weights = np.ones(4)

    measured = MEASUREMENT._tail_measurements(values, weights, (10,))

    assert measured["disclosure_suppressed"] is True
    assert measured["carriers"] is None
    assert measured["nonzero_share"] is None
    assert measured["top_shares"] == {}


def test_column_at_the_threshold_reports_its_statistics() -> None:
    values = np.concatenate([np.ones(10), np.zeros(5)])
    weights = np.ones(values.size)

    measured = MEASUREMENT._tail_measurements(
        values,
        weights,
        (10,),
        minimum_count=10,
    )

    assert measured["disclosure_suppressed"] is False
    assert measured["carriers"] == 10
    assert measured["top_shares"] == {"10": 1.0}


def test_concentration_is_still_measurable_above_the_threshold() -> None:
    """Suppression must not blind the recorder to a real #462 signature."""

    values = np.concatenate([[1_000.0], np.ones(99)])
    weights = np.ones(values.size)

    measured = MEASUREMENT._tail_measurements(values, weights, (10,))

    assert measured["disclosure_suppressed"] is False
    assert measured["top_shares"]["10"] > 0.9


def test_top_k_below_the_minimum_count_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_uk_weighted_integrity_baselines.py",
            "--h5",
            f"fixture={tmp_path / 'fixture.h5'}",
            "--top-k",
            "1,100",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(SystemExit, match="below --sdc-minimum-count"):
        MEASUREMENT.main()


def test_minimum_count_below_the_standard_floor_is_refused(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_uk_weighted_integrity_baselines.py",
            "--h5",
            f"fixture={tmp_path / 'fixture.h5'}",
            "--sdc-minimum-count",
            "2",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(SystemExit, match="at least 3"):
        MEASUREMENT.main()


def test_default_minimum_count_follows_the_secondary_disclosure_advice() -> None:
    assert MEASUREMENT.DEFAULT_SDC_MINIMUM_COUNT == 10
    assert all(k >= 10 for k in MEASUREMENT.DEFAULT_TOP_K_GRID)
