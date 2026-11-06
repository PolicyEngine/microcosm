"""Schedule E line 32 against a reference built before the entity split.

microcosm#1261 split line 32 between ``partnership_income`` and
``s_corp_income``. A reference built before it holds the whole line in
``partnership_income``, so the input-mass gate counts both candidate columns
against that one until the reference carries the split.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.build.gates import input_mass_parity_gate
from microcosm.build.us_runtime import release_gate_dry_run as dry
from microcosm.build.us_runtime.input_mass import us_line_32_on_reference_terms
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

FLOOR = 1e9
TOLERANCE = 0.5

# Weighted totals from release attempt
# populace-us-2024-65671e1-9434c504ba2a-20261103T095117Z: its export and its
# reference, populace_us_2024.h5, which carries no s_corp_income record.
REFERENCE = {"partnership_income": 847.756e9, "s_corp_income": 0.0}
EXPORT = {"partnership_income": 391.947e9, "s_corp_income": 664.150e9}

_mass = st.floats(min_value=-2e12, max_value=2e12, allow_nan=False)
_share = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _gate(candidate, reference):
    return input_mass_parity_gate(
        us_line_32_on_reference_terms(
            candidate, reference, minimum_reference_total=FLOOR
        ),
        reference,
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
    )


@pytest.fixture(scope="module")
def builder():
    path = _TEST_PATHS.repository / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_line_32", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_export_that_keeps_line_32_passes_a_pre_split_reference() -> None:
    alone = input_mass_parity_gate(
        EXPORT,
        REFERENCE,
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
    )
    assert not alone.passed
    assert alone.failures[0].startswith("partnership_income:")

    gate = _gate(EXPORT, REFERENCE)
    assert gate.passed, gate.failures
    assert gate.details["worst_drifts"]["partnership_income"] == pytest.approx(
        (391.947 + 664.150 - 847.756) / 847.756
    )


def test_a_pre_split_reference_still_refuses_lost_line_32_mass() -> None:
    gate = _gate({"partnership_income": 2e11, "s_corp_income": 1e11}, REFERENCE)
    assert not gate.passed
    assert gate.failures[0].startswith("partnership_income:")


def test_a_reference_with_the_split_compares_each_column_alone() -> None:
    reference = {"partnership_income": 450e9, "s_corp_income": 600e9}
    gate = _gate({"partnership_income": 100e9, "s_corp_income": 400e9}, reference)
    assert not gate.passed
    assert [line.split(":", 1)[0] for line in gate.failures] == [
        "partnership_income",
    ]


def test_a_candidate_without_partnership_income_stays_refused() -> None:
    candidate = {"s_corp_income": 5e11}
    assert "partnership_income" not in us_line_32_on_reference_terms(
        candidate, REFERENCE, minimum_reference_total=FLOOR
    )
    gate = _gate(candidate, REFERENCE)
    assert not gate.passed
    assert "absent" in gate.failures[0]


def test_a_candidate_without_s_corp_income_keeps_its_partnership_income() -> None:
    assert us_line_32_on_reference_terms(
        {"partnership_income": 8e11}, REFERENCE, minimum_reference_total=FLOOR
    ) == {"partnership_income": 8e11}


@given(
    partnership=_mass,
    s_corp=_mass,
    other=_mass,
    reference_s_corp=_mass,
)
def test_only_partnership_income_moves_and_only_on_a_pre_split_reference(
    partnership: float, s_corp: float, other: float, reference_s_corp: float
) -> None:
    candidate = {
        "partnership_income": partnership,
        "s_corp_income": s_corp,
        "other": other,
    }
    before = dict(candidate)
    totals = us_line_32_on_reference_terms(
        candidate,
        {"s_corp_income": reference_s_corp},
        minimum_reference_total=FLOOR,
    )
    assert candidate == before
    assert totals["s_corp_income"] == s_corp
    assert totals["other"] == other
    expected = partnership + s_corp if abs(reference_s_corp) <= FLOOR else partnership
    assert totals["partnership_income"] == expected


@given(
    line_32=st.floats(min_value=1e9, max_value=2e12, allow_nan=False),
    partnership_share=_share,
)
def test_the_split_never_changes_the_verdict_on_a_pre_split_reference(
    line_32: float, partnership_share: float
) -> None:
    # Moving income between the entity columns keeps the total to rounding, so
    # the gate's verdict tracks the total alone.
    partnership = line_32 * partnership_share
    candidate = {
        "partnership_income": partnership,
        "s_corp_income": line_32 - partnership,
    }
    drift = abs(line_32 - REFERENCE["partnership_income"]) / abs(
        REFERENCE["partnership_income"]
    )
    if abs(drift - TOLERANCE) < 1e-9:
        return
    assert _gate(candidate, REFERENCE).passed is (drift <= TOLERANCE)


@given(partnership=_mass, s_corp=_mass, reference_s_corp=_mass)
def test_the_dry_run_preview_agrees_with_the_gate(
    partnership: float, s_corp: float, reference_s_corp: float
) -> None:
    # The preview raises when its own arithmetic disagrees with the gate's lines.
    reference = {"partnership_income": 8e11, "s_corp_income": reference_s_corp}
    candidate = {"partnership_income": partnership, "s_corp_income": s_corp}
    gate = _gate(candidate, reference)
    check = dry.export_input_mass_check(
        gate_failures=gate.failures,
        gate_details=gate.details,
        candidate_totals=us_line_32_on_reference_terms(
            candidate, reference, minimum_reference_total=FLOOR
        ),
        reference_totals=reference,
        candidate_nonzero_shares={"partnership_income": 0.02, "s_corp_income": 0.02},
        reviewed_exclusions={},
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
        margin=0.1,
        allow_drift=False,
        reference_is_candidate=False,
        reference_label="reference.h5",
    )
    rows = {row["column"]: row for row in check.rows}
    assert "partnership_income" in rows


def _sentinel_totals(monkeypatch, builder, **totals_by_label):
    frames = {label: SimpleNamespace(label=label) for label in totals_by_label}
    monkeypatch.setattr(builder, "_engine_input_variables", lambda: ())
    monkeypatch.setattr(
        builder,
        "us_input_mass_totals",
        lambda frame, columns=None: totals_by_label[frame.label],
    )
    return frames


def test_the_release_export_gate_reads_line_32_on_the_reference_terms(
    builder, monkeypatch
) -> None:
    frames = _sentinel_totals(
        monkeypatch, builder, export=EXPORT, base=EXPORT, reference=REFERENCE
    )
    gate = builder._export_input_mass_gate(
        frames["export"],
        frames["base"],
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
        reference_frame=frames["reference"],
        reference_name="populace_us_2024.h5",
    )
    assert gate.passed, gate.failures


def test_the_release_export_gate_refuses_lost_line_32_mass(
    builder, monkeypatch
) -> None:
    lost = {"partnership_income": 2e11, "s_corp_income": 1e11}
    frames = _sentinel_totals(
        monkeypatch, builder, export=lost, base=EXPORT, reference=REFERENCE
    )
    gate = builder._export_input_mass_gate(
        frames["export"],
        frames["base"],
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
        reference_frame=frames["reference"],
        reference_name="populace_us_2024.h5",
    )
    assert not gate.passed
    assert gate.failures[0].startswith("partnership_income:")


def test_the_release_base_gate_reads_line_32_on_the_reference_terms(
    builder, monkeypatch, tmp_path
) -> None:
    frames = _sentinel_totals(monkeypatch, builder, base=EXPORT, reference=REFERENCE)
    monkeypatch.setattr(builder, "load_us_frame", lambda path: frames["reference"])
    gate = builder._input_mass_reference_gate(
        frames["base"],
        reference_h5=tmp_path / "populace_us_2024.h5",
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
    )
    assert gate is not None
    assert gate.passed, gate.failures
