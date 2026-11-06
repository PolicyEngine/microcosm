"""Schedule E line 32 against a reference built before the entity split.

microcosm#1261 split line 32 between ``partnership_income`` and
``s_corp_income``. A reference built before it holds partnership and S
corporation income together in ``partnership_income`` and zero in
``s_corp_income``, so the input-mass gate counts both candidate columns against
that one until the reference holds any ``s_corp_income``.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from microcosm.build.gates import input_mass_parity_gate
from microcosm.build.us_runtime import release_gate_dry_run as dry
from microcosm.build.us_runtime.input_mass import (
    us_input_mass_parity_gate,
    us_line_32_on_reference_terms,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

FLOOR = 1e9
TOLERANCE = 0.5
BOTH_COLUMNS = ["partnership_income", "s_corp_income"]

# Weighted totals from release attempt
# populace-us-2024-65671e1-9434c504ba2a-20261103T095117Z: its export and its
# reference, populace_us_2024.h5, which carries no s_corp_income record.
REFERENCE = {"partnership_income": 847.756e9, "s_corp_income": 0.0}
EXPORT = {"partnership_income": 391.947e9, "s_corp_income": 664.150e9}

# The rule tests for exact zeros, so every draw can land on one.
_mass = st.floats(min_value=-2e12, max_value=2e12) | st.just(0.0)
_REFUSING_VERDICTS = {"zero_mass", "zero_mass_by_cancellation", "out_of_band"}


def _gate(candidate, reference, floor=FLOOR):
    return us_input_mass_parity_gate(
        candidate,
        reference,
        relative_tolerance=TOLERANCE,
        minimum_reference_total=floor,
    )


def _preview(candidate, reference, *, gate_failures=None, partnership_share=0.02):
    gate = _gate(candidate, reference)
    return dry.export_input_mass_check(
        gate_failures=gate.failures if gate_failures is None else gate_failures,
        gate_details=gate.details,
        candidate_totals=us_line_32_on_reference_terms(candidate, reference),
        reference_totals=reference,
        candidate_nonzero_shares={
            "partnership_income": partnership_share,
            "s_corp_income": 0.02,
        },
        reviewed_exclusions={},
        relative_tolerance=TOLERANCE,
        minimum_reference_total=FLOOR,
        margin=0.1,
        allow_drift=False,
        reference_is_candidate=False,
        reference_label="reference.h5",
    )


def _partnership_row(check):
    return {row["column"]: row for row in check.rows}["partnership_income"]


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
    assert gate.details["partnership_income_counts"] == BOTH_COLUMNS


def test_a_pre_split_reference_still_refuses_lost_line_32_mass() -> None:
    gate = _gate({"partnership_income": 2e11, "s_corp_income": 1e11}, REFERENCE)
    assert not gate.passed
    assert gate.failures[0].startswith("partnership_income:")


def test_a_zeroed_partnership_column_stays_refused() -> None:
    # S corporation income alone lands inside the reference's band, so counting
    # it here would pass a build that lost the whole column.
    alone = EXPORT["s_corp_income"] / REFERENCE["partnership_income"] - 1
    assert abs(alone) <= TOLERANCE

    candidate = {"partnership_income": 0.0, "s_corp_income": EXPORT["s_corp_income"]}
    gate = _gate(candidate, REFERENCE)
    assert not gate.passed
    assert "mass is zero" in gate.failures[0]
    assert "partnership_income_counts" not in gate.details


def test_a_wrong_split_passes_a_pre_split_reference() -> None:
    # Intended limit: a reference without the split can't show how a candidate
    # divides the income, so the gate reads only the total.
    assert _gate(
        {"partnership_income": -1e12, "s_corp_income": 1.9e12}, REFERENCE
    ).passed
    assert _gate({"partnership_income": 1.0, "s_corp_income": 8e11}, REFERENCE).passed


def test_a_reference_with_the_split_compares_each_column_alone() -> None:
    reference = {"partnership_income": 450e9, "s_corp_income": 600e9}
    gate = _gate({"partnership_income": 100e9, "s_corp_income": 400e9}, reference)
    assert not gate.passed
    assert [line.split(":", 1)[0] for line in gate.failures] == [
        "partnership_income",
    ]
    assert "partnership_income_counts" not in gate.details


@pytest.mark.parametrize("reference_s_corp", [1.0, -1.0, FLOOR, -FLOOR, 0.9 * FLOOR])
def test_any_s_corp_income_in_the_reference_ends_the_rule(
    reference_s_corp: float,
) -> None:
    # The gate skips a total at or under its floor, and that floor says nothing
    # about what the reference's columns hold.
    reference = {"partnership_income": 847.756e9, "s_corp_income": reference_s_corp}
    gate = _gate(EXPORT, reference)
    assert not gate.passed
    assert gate.failures[0].startswith("partnership_income:")


@pytest.mark.parametrize(
    ("partnership", "s_corp"),
    [(1.5e9, 1e9), (1.1e9, 0.9e9), (1.5e9, 1.00001e9), (1.5e9, -1e9)],
)
def test_a_frame_with_small_s_corp_income_passes_against_itself(
    partnership: float, s_corp: float
) -> None:
    totals = {"partnership_income": partnership, "s_corp_income": s_corp}
    gate = _gate(totals, totals)
    assert gate.passed, gate.failures


def test_a_candidate_without_partnership_income_stays_refused() -> None:
    candidate = {"s_corp_income": 5e11}
    assert "partnership_income" not in us_line_32_on_reference_terms(
        candidate, REFERENCE
    )
    gate = _gate(candidate, REFERENCE)
    assert not gate.passed
    assert "absent" in gate.failures[0]


def test_a_candidate_without_s_corp_income_keeps_its_partnership_income() -> None:
    assert us_line_32_on_reference_terms({"partnership_income": 8e11}, REFERENCE) == {
        "partnership_income": 8e11
    }


def test_a_reference_without_the_column_reads_as_pre_split() -> None:
    totals = us_line_32_on_reference_terms(EXPORT, {"partnership_income": 8e11})
    assert totals["partnership_income"] == sum(EXPORT.values())


@given(partnership=_mass, s_corp=_mass, other=_mass, reference_s_corp=_mass)
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
        candidate, {"s_corp_income": reference_s_corp}
    )
    assert candidate == before
    assert totals["s_corp_income"] == s_corp
    assert totals["other"] == other
    counts_both = partnership != 0.0 and reference_s_corp == 0.0
    expected = partnership + s_corp if counts_both else partnership
    assert totals["partnership_income"] == expected


@given(partnership=_mass, s_corp=_mass, floor=st.sampled_from([0.0, FLOOR]))
def test_a_frame_compared_with_itself_never_drifts(
    partnership: float, s_corp: float, floor: float
) -> None:
    totals = {"partnership_income": partnership, "s_corp_income": s_corp}
    gate = _gate(totals, totals, floor=floor)
    assert gate.passed, gate.failures
    assert all(drift == 0.0 for drift in gate.details["worst_drifts"].values())


@given(line_32=st.floats(min_value=1e9, max_value=2e12), partnership=_mass)
def test_the_split_never_changes_the_verdict_on_a_pre_split_reference(
    line_32: float, partnership: float
) -> None:
    # Moving signed income between the entity columns keeps the total to
    # rounding, so the gate's verdict tracks the total alone. A zeroed
    # partnership_income column has its own refusal, tested above.
    assume(partnership != 0.0)
    s_corp = line_32 - partnership
    reference = REFERENCE["partnership_income"]
    drift = abs(partnership + s_corp - reference) / abs(reference)
    assume(abs(drift - TOLERANCE) >= 1e-9)
    candidate = {"partnership_income": partnership, "s_corp_income": s_corp}
    assert _gate(candidate, REFERENCE).passed is (drift <= TOLERANCE)


@given(partnership=_mass, s_corp=_mass, reference_s_corp=_mass)
def test_the_dry_run_preview_agrees_with_the_gate(
    partnership: float, s_corp: float, reference_s_corp: float
) -> None:
    reference = {"partnership_income": 8e11, "s_corp_income": reference_s_corp}
    candidate = {"partnership_income": partnership, "s_corp_income": s_corp}
    gate = _gate(candidate, reference)
    refused = any(line.startswith("partnership_income:") for line in gate.failures)
    row = _partnership_row(_preview(candidate, reference))
    assert (row["verdict"] in _REFUSING_VERDICTS) is refused


def test_the_preview_raises_when_the_gate_passes_what_it_reads_as_failing() -> None:
    lost = {"partnership_income": 2e11, "s_corp_income": 1e11}
    with pytest.raises(ValueError, match="the release's gate passes it"):
        _preview(lost, REFERENCE, gate_failures=())


def test_the_preview_raises_when_the_gate_fails_what_it_reads_in_band() -> None:
    with pytest.raises(ValueError, match="the preview reads it in band"):
        _preview(EXPORT, REFERENCE, gate_failures=("partnership_income: refused",))


def test_the_preview_refuses_a_column_with_no_partnership_record_for_certain() -> None:
    # No weights can give these records partnership_income, so the rule never
    # counts their S corporation income.
    candidate = {"partnership_income": 0.0, "s_corp_income": EXPORT["s_corp_income"]}
    row = _partnership_row(_preview(candidate, REFERENCE, partnership_share=0.0))
    assert (row["verdict"], row["status"], row["certain"]) == (
        "zero_mass",
        "FAIL",
        True,
    )


def test_the_preview_reads_cancelling_partnership_records_as_at_risk() -> None:
    # Other weights can leave partnership_income nonzero, and the rule then
    # counts both columns.
    candidate = {"partnership_income": 0.0, "s_corp_income": EXPORT["s_corp_income"]}
    row = _partnership_row(_preview(candidate, REFERENCE))
    assert (row["verdict"], row["status"], row["certain"]) == (
        "zero_mass_by_cancellation",
        "AT_RISK",
        False,
    )


def test_the_preview_reads_records_that_cancel_across_the_columns_as_at_risk() -> None:
    # Intended: if every partnership record has an opposite S corporation record,
    # no weights can rescue the total, so FAIL would hold too. The preview sees
    # one share per column and can't tell, and the gate still refuses at export.
    candidate = {"partnership_income": 5e11, "s_corp_income": -5e11}
    assert "mass is zero" in _gate(candidate, REFERENCE).failures[0]
    row = _partnership_row(_preview(candidate, REFERENCE))
    assert (row["verdict"], row["status"], row["certain"]) == (
        "zero_mass_by_cancellation",
        "AT_RISK",
        False,
    )


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
    assert gate.details["partnership_income_counts"] == BOTH_COLUMNS


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
    assert gate.details["partnership_income_counts"] == BOTH_COLUMNS
