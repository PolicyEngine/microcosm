"""Schedule E line 32 against a reference built before the entity split.

microcosm#1261 split line 32 between ``partnership_income`` and
``s_corp_income``. A reference built before it holds partnership and S
corporation income together in ``partnership_income`` and no ``s_corp_income``
record, so the input-mass gate counts both candidate columns against that one
until any reference record carries ``s_corp_income``. The rule reads which
records carry each column, because a weighted total can't tell records that
cancel from no records.
"""

from __future__ import annotations

import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.gates import input_mass_parity_gate
from microcosm.build.us_runtime import release_gate_dry_run as dry
from microcosm.build.us_runtime.input_mass import (
    US_LINE_32_COLUMNS,
    UsLine32Carriers,
    us_input_mass_parity_gate,
    us_input_mass_totals,
    us_line_32_carriers,
    us_line_32_counts_both,
    us_line_32_on_reference_terms,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
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

# That pair's carriers: weighted export records hold partnership income, and no
# reference record holds S corporation income.
PRE_SPLIT = UsLine32Carriers(
    candidate_partnership_income=True, reference_s_corp_income=False
)
# A reference record holds S corporation income.
SPLIT = UsLine32Carriers(
    candidate_partnership_income=True, reference_s_corp_income=True
)
# No candidate record with weight holds partnership income.
LOST = UsLine32Carriers(
    candidate_partnership_income=False, reference_s_corp_income=False
)

# The rule tests for exact zeros, so every draw can land on one.
_mass = st.floats(min_value=-2e12, max_value=2e12) | st.just(0.0)
_REFUSING_VERDICTS = {"zero_mass", "zero_mass_by_cancellation", "out_of_band"}

# Small frames: amounts that cancel in pairs or go missing, and weights that
# include zero.
_amount = st.sampled_from([0.0, 1.0, -1.0, 2.5, -2.5, 3e9, -3e9, math.nan])
_weight = st.sampled_from([0.0, 0.5, 1.0, 2.0])


def _frame(partnership, s_corp, weights):
    """A US frame of one-person households; ``None`` leaves a column out."""
    ids = np.arange(1, len(weights) + 1, dtype=np.int64)
    person = {"person_id": ids}
    for entity in US_SCHEMA.group_entities:
        person[US_SCHEMA.membership_column(entity)] = ids
    for column, values in zip(US_LINE_32_COLUMNS, (partnership, s_corp), strict=True):
        if values is not None:
            person[column] = np.asarray(values, dtype=np.float64)
    tables = {"person": pd.DataFrame(person)}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({US_SCHEMA.id_column(entity): ids})
    household = Weights(np.asarray(weights, dtype=np.float64), WeightKind.DESIGN)
    return Frame(tables, US_SCHEMA, {"household": household})


@st.composite
def _frames(draw):
    size = draw(st.integers(min_value=1, max_value=4))
    column = st.none() | st.lists(_amount, min_size=size, max_size=size)
    # A frame refuses weights that are all zero.
    weights = st.lists(_weight, min_size=size, max_size=size).filter(any)
    return draw(column), draw(column), draw(weights)


def _gate(candidate, reference, carriers=PRE_SPLIT, floor=FLOOR):
    return us_input_mass_parity_gate(
        candidate,
        reference,
        carriers=carriers,
        relative_tolerance=TOLERANCE,
        minimum_reference_total=floor,
    )


def _frame_gate(candidate, reference, floor=0.0):
    return _gate(
        us_input_mass_totals(candidate),
        us_input_mass_totals(reference),
        us_line_32_carriers(candidate, reference),
        floor=floor,
    )


def _preview(
    candidate,
    reference,
    carriers=PRE_SPLIT,
    *,
    gate_failures=None,
    partnership_share=0.02,
):
    gate = _gate(candidate, reference, carriers)
    return dry.export_input_mass_check(
        gate_failures=gate.failures if gate_failures is None else gate_failures,
        gate_details=gate.details,
        candidate_totals=us_line_32_on_reference_terms(candidate, reference, carriers),
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
    gate = _gate(candidate, REFERENCE, LOST)
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
    gate = _gate(
        {"partnership_income": 100e9, "s_corp_income": 400e9}, reference, SPLIT
    )
    assert not gate.passed
    assert [line.split(":", 1)[0] for line in gate.failures] == [
        "partnership_income",
    ]
    assert "partnership_income_counts" not in gate.details


@pytest.mark.parametrize(
    "reference_s_corp", [1.0, -1.0, FLOOR, -FLOOR, 0.9 * FLOOR, 0.0]
)
def test_any_s_corp_record_in_the_reference_ends_the_rule(
    reference_s_corp: float,
) -> None:
    # The gate skips a total at or under its floor, and records that cancel
    # leave a zero total; neither says what the reference's columns hold.
    reference = {"partnership_income": 847.756e9, "s_corp_income": reference_s_corp}
    gate = _gate(EXPORT, reference, SPLIT)
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
    gate = _gate(totals, totals, SPLIT)
    assert gate.passed, gate.failures


def test_a_candidate_without_partnership_income_stays_refused() -> None:
    # Totals that don't measure the column keep it out, whatever the carriers say.
    candidate = {"s_corp_income": 5e11}
    assert "partnership_income" not in us_line_32_on_reference_terms(
        candidate, REFERENCE, PRE_SPLIT
    )
    gate = _gate(candidate, REFERENCE, LOST)
    assert not gate.passed
    assert "absent" in gate.failures[0]


def test_a_candidate_without_s_corp_income_keeps_its_partnership_income() -> None:
    assert us_line_32_on_reference_terms(
        {"partnership_income": 8e11}, REFERENCE, PRE_SPLIT
    ) == {"partnership_income": 8e11}


def test_carriers_that_contradict_the_totals_raise() -> None:
    # A record carries every nonzero total, so a mismatch means the carriers
    # came from other frames than the totals did.
    with pytest.raises(ValueError, match="no candidate record with nonzero weight"):
        _gate(EXPORT, REFERENCE, LOST)
    split = {"partnership_income": 450e9, "s_corp_income": 600e9}
    with pytest.raises(ValueError, match="no reference record carries it"):
        us_line_32_on_reference_terms(EXPORT, split, PRE_SPLIT)


def test_a_non_finite_total_reaches_the_gates_own_refusal() -> None:
    candidate = {"partnership_income": math.nan, "s_corp_income": 1.0}
    with pytest.raises(ValueError, match="must be finite"):
        _gate(candidate, REFERENCE, LOST)


def test_carriers_read_records_rather_than_totals() -> None:
    # A split reference whose S corporation records cancel to a zero total.
    split = _frame([1, 1], [2, -2], [1, 1])
    assert us_input_mass_totals(split)["s_corp_income"] == 0.0
    assert us_line_32_carriers(split, split) == SPLIT
    # Pre-split references: every s_corp_income amount is zero, or no column.
    for reference in (_frame([2, 0], [0, 0], [1, 1]), _frame([2], None, [1])):
        assert us_line_32_carriers(split, reference) == PRE_SPLIT


def test_a_reference_s_corp_record_counts_at_any_weight() -> None:
    reference = _frame([2, 0], [0, 3], [1, 0])
    assert us_input_mass_totals(reference)["s_corp_income"] == 0.0
    assert us_line_32_carriers(reference, reference).reference_s_corp_income


def test_a_candidate_partnership_record_counts_only_under_weight() -> None:
    def carries(candidate):
        return us_line_32_carriers(candidate, candidate).candidate_partnership_income

    assert carries(_frame([1, -1], [0.5, 0.5], [1, 1]))
    assert not carries(_frame([1, 0], [0, 0.8], [0, 1]))
    assert not carries(_frame(None, [1], [1]))


@pytest.mark.parametrize("floor", [0.0, 1.0])
@pytest.mark.parametrize("scale", [1.0, 1e9])
def test_a_reference_whose_s_corp_records_cancel_compares_each_column(
    scale: float, floor: float
) -> None:
    # Its totals match a pre-split reference's: the S corporation records net
    # to zero.
    reference = _frame([scale, scale], [2 * scale, -2 * scale], [1, 1])

    # Its own records reweighted keep partnership income, which counting both
    # columns would read as +100%.
    reweighted = _frame([scale, scale], [2 * scale, -2 * scale], [1.5, 0.5])
    assert us_input_mass_totals(reweighted) == {
        "partnership_income": 2 * scale,
        "s_corp_income": 2 * scale,
    }
    gate = _frame_gate(reweighted, reference, floor=floor * scale)
    assert gate.passed, gate.failures
    assert "partnership_income_counts" not in gate.details

    # A 75% partnership loss, which counting both columns would pass.
    lost = _frame([0.25 * scale, 0.25 * scale], [3 * scale, -1.5 * scale], [1, 1])
    gate = _frame_gate(lost, reference, floor=floor * scale)
    assert [line.split(":", 1)[0] for line in gate.failures] == [
        "partnership_income",
    ]
    assert gate.details["worst_drifts"]["partnership_income"] == pytest.approx(-0.75)


@pytest.mark.parametrize(("scale", "floor"), [(1.0, 0.0), (2e9, FLOOR)])
def test_cancelling_partnership_records_count_both_columns(
    scale: float, floor: float
) -> None:
    reference = _frame([scale], [0], [1])
    candidate = _frame([scale, -scale], [0.5 * scale, 0.5 * scale], [1, 1])
    assert us_input_mass_totals(candidate)["partnership_income"] == 0.0
    gate = _frame_gate(candidate, reference, floor=floor)
    assert gate.passed, gate.failures
    assert gate.details["partnership_income_counts"] == BOTH_COLUMNS
    assert gate.details["worst_drifts"]["partnership_income"] == 0.0


def test_partnership_records_without_weight_stay_refused() -> None:
    # S corporation income alone reads -20%, inside the band.
    reference = _frame([1], [0], [1])
    candidate = _frame([1, 0], [0, 0.8], [0, 1])
    gate = _frame_gate(candidate, reference)
    assert not gate.passed
    assert "mass is zero" in gate.failures[0]
    assert "partnership_income_counts" not in gate.details


@given(
    partnership=_mass,
    s_corp=_mass,
    other=_mass,
    reference_s_corp=_mass,
    candidate_carries=st.booleans(),
    reference_carries=st.booleans(),
)
def test_only_partnership_income_moves_and_only_on_a_pre_split_reference(
    partnership: float,
    s_corp: float,
    other: float,
    reference_s_corp: float,
    candidate_carries: bool,
    reference_carries: bool,
) -> None:
    # A record carries every nonzero total.
    carriers = UsLine32Carriers(
        candidate_partnership_income=candidate_carries or partnership != 0.0,
        reference_s_corp_income=reference_carries or reference_s_corp != 0.0,
    )
    candidate = {
        "partnership_income": partnership,
        "s_corp_income": s_corp,
        "other": other,
    }
    reference = {"s_corp_income": reference_s_corp}
    before = dict(candidate)
    totals = us_line_32_on_reference_terms(candidate, reference, carriers)
    assert candidate == before
    assert totals["s_corp_income"] == s_corp
    assert totals["other"] == other
    counts_both = (
        carriers.candidate_partnership_income and not carriers.reference_s_corp_income
    )
    assert us_line_32_counts_both(candidate, reference, carriers) is counts_both
    expected = partnership + s_corp if counts_both else partnership
    assert totals["partnership_income"] == expected


@given(
    partnership=_mass,
    s_corp=_mass,
    floor=st.sampled_from([0.0, FLOOR]),
    partnership_carries=st.booleans(),
    s_corp_carries=st.booleans(),
)
def test_totals_compared_with_themselves_never_drift(
    partnership: float,
    s_corp: float,
    floor: float,
    partnership_carries: bool,
    s_corp_carries: bool,
) -> None:
    carriers = UsLine32Carriers(
        candidate_partnership_income=partnership_carries or partnership != 0.0,
        reference_s_corp_income=s_corp_carries or s_corp != 0.0,
    )
    totals = {"partnership_income": partnership, "s_corp_income": s_corp}
    gate = _gate(totals, totals, carriers, floor=floor)
    assert gate.passed, gate.failures
    assert all(drift == 0.0 for drift in gate.details["worst_drifts"].values())


@settings(deadline=None)
@given(frame=_frames(), floor=st.sampled_from([0.0, 1.0]))
def test_a_frame_compared_with_itself_never_drifts(frame, floor: float) -> None:
    built = _frame(*frame)
    gate = _frame_gate(built, built, floor=floor)
    assert gate.passed, gate.failures
    assert all(drift == 0.0 for drift in gate.details["worst_drifts"].values())


@settings(deadline=None)
@given(candidate=_frames(), reference=_frames())
def test_carriers_match_a_record_by_record_reading(candidate, reference) -> None:
    # A missing amount totals as zero, so it holds nothing.
    def holds(amount):
        return not math.isnan(amount) and amount != 0

    partnership, _, weights = candidate
    _, s_corp, _ = reference
    carriers = us_line_32_carriers(_frame(*candidate), _frame(*reference))
    assert carriers.candidate_partnership_income is (
        partnership is not None
        and any(holds(p) and w != 0 for p, w in zip(partnership, weights, strict=True))
    )
    assert carriers.reference_s_corp_income is (
        s_corp is not None and any(holds(s) for s in s_corp)
    )


@settings(deadline=None)
@given(candidate=_frames(), reference=_frames())
def test_carriers_from_the_frames_never_contradict_their_totals(
    candidate, reference
) -> None:
    # The rule raises on a nonzero total without a carrier; real frames never
    # trip it.
    candidate_frame, reference_frame = _frame(*candidate), _frame(*reference)
    us_line_32_counts_both(
        us_input_mass_totals(candidate_frame),
        us_input_mass_totals(reference_frame),
        us_line_32_carriers(candidate_frame, reference_frame),
    )


@given(line_32=st.floats(min_value=1e9, max_value=2e12), partnership=_mass)
def test_the_split_never_changes_the_verdict_on_a_pre_split_reference(
    line_32: float, partnership: float
) -> None:
    # Moving signed income between the entity columns keeps the total to
    # rounding, so the gate's verdict tracks the total alone while a weighted
    # record carries partnership income, even one whose amounts net to zero.
    s_corp = line_32 - partnership
    reference = REFERENCE["partnership_income"]
    drift = abs(partnership + s_corp - reference) / abs(reference)
    assume(abs(drift - TOLERANCE) >= 1e-9)
    candidate = {"partnership_income": partnership, "s_corp_income": s_corp}
    assert _gate(candidate, REFERENCE).passed is (drift <= TOLERANCE)


@given(
    partnership=_mass,
    s_corp=_mass,
    reference_s_corp=_mass,
    candidate_carries=st.booleans(),
    reference_carries=st.booleans(),
    unweighted_share=st.sampled_from([0.0, 0.02]),
)
def test_the_dry_run_preview_agrees_with_the_gate(
    partnership: float,
    s_corp: float,
    reference_s_corp: float,
    candidate_carries: bool,
    reference_carries: bool,
    unweighted_share: float,
) -> None:
    carriers = UsLine32Carriers(
        candidate_partnership_income=candidate_carries or partnership != 0.0,
        reference_s_corp_income=reference_carries or reference_s_corp != 0.0,
    )
    # A weighted carrier is a record, and the record share also counts records
    # at zero weight.
    share = 0.02 if carriers.candidate_partnership_income else unweighted_share
    reference = {"partnership_income": 8e11, "s_corp_income": reference_s_corp}
    candidate = {"partnership_income": partnership, "s_corp_income": s_corp}
    gate = _gate(candidate, reference, carriers)
    refused = any(line.startswith("partnership_income:") for line in gate.failures)
    row = _partnership_row(
        _preview(candidate, reference, carriers, partnership_share=share)
    )
    assert (row["verdict"] in _REFUSING_VERDICTS) is refused


def test_the_preview_raises_when_the_gate_passes_what_it_reads_as_failing() -> None:
    lost = {"partnership_income": 2e11, "s_corp_income": 1e11}
    with pytest.raises(ValueError, match="the release's gate passes it"):
        _preview(lost, REFERENCE, gate_failures=())


def test_the_preview_raises_when_the_gate_fails_what_it_reads_in_band() -> None:
    with pytest.raises(ValueError, match="the preview reads it in band"):
        _preview(EXPORT, REFERENCE, gate_failures=("partnership_income: refused",))


def test_the_preview_refuses_a_column_with_no_partnership_record_for_certain() -> None:
    # No weights can give these records partnership income, so the rule never
    # counts their S corporation income.
    candidate = {"partnership_income": 0.0, "s_corp_income": EXPORT["s_corp_income"]}
    row = _partnership_row(_preview(candidate, REFERENCE, LOST, partnership_share=0.0))
    assert (row["verdict"], row["status"], row["certain"]) == (
        "zero_mass",
        "FAIL",
        True,
    )


def test_the_preview_counts_both_columns_for_cancelling_partnership_records() -> None:
    # Weighted records carry partnership income that nets to zero, so the gate
    # compares the combined line, which keeps 78% of the reference.
    candidate = {"partnership_income": 0.0, "s_corp_income": EXPORT["s_corp_income"]}
    assert _gate(candidate, REFERENCE).passed
    row = _partnership_row(_preview(candidate, REFERENCE))
    assert row["mass_at_base_weights"] == EXPORT["s_corp_income"]
    assert (row["verdict"], row["status"]) == ("in_band", "PASS")


def test_the_preview_reads_partnership_records_without_weight_as_at_risk() -> None:
    # Records at zero base weight can gain weight in the solve, and the rule
    # then counts both columns, so the gate's refusal here isn't certain.
    candidate = {"partnership_income": 0.0, "s_corp_income": EXPORT["s_corp_income"]}
    assert "mass is zero" in _gate(candidate, REFERENCE, LOST).failures[0]
    row = _partnership_row(_preview(candidate, REFERENCE, LOST))
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


def _sentinel_frames(monkeypatch, builder, *, carriers_from, **totals_by_label):
    frames = {label: SimpleNamespace(label=label) for label in totals_by_label}
    monkeypatch.setattr(builder, "_engine_input_variables", lambda: ())
    monkeypatch.setattr(
        builder,
        "us_input_mass_totals",
        lambda frame, columns=None: totals_by_label[frame.label],
    )

    def carriers(candidate, reference):
        # The gate must read carriers from the frames whose totals it compares.
        assert (candidate.label, reference.label) == carriers_from
        return PRE_SPLIT

    monkeypatch.setattr(builder, "us_line_32_carriers", carriers)
    return frames


def test_the_release_export_gate_reads_line_32_on_the_reference_terms(
    builder, monkeypatch
) -> None:
    frames = _sentinel_frames(
        monkeypatch,
        builder,
        carriers_from=("export", "reference"),
        export=EXPORT,
        base=EXPORT,
        reference=REFERENCE,
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
    frames = _sentinel_frames(
        monkeypatch,
        builder,
        carriers_from=("export", "reference"),
        export=lost,
        base=EXPORT,
        reference=REFERENCE,
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
    frames = _sentinel_frames(
        monkeypatch,
        builder,
        carriers_from=("base", "reference"),
        base=EXPORT,
        reference=REFERENCE,
    )
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
