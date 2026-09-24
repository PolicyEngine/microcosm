"""Cross-register consistency: the eCPS parity known-gap leg (microcosm #377).

The default path used to load the parity known-gap register as
``ParityKnownGap`` objects and intersect them with column-name sets, so the
parity leg could never report a contradiction. These tests pin that the
register reaches the check by name, and that a gap filed under the pinned
reference's historical spelling resolves through ``REFERENCE_ECPS_LAYER_RENAMES``
onto the live column the parity gate exempts (gate peer P2 on
PolicyEngine/microcosm#994).
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime.parity_reference import load_ecps_parity_known_gaps
from microcosm.build.us_runtime.register_consistency import (
    us_register_consistency_gate,
    us_register_contradictions,
)

_PARITY_GAP_PHRASE = "excused as absent/degenerate by eCPS parity known gaps"
_WIC_SPELLINGS = ("would_claim_wic", "takes_up_wic_if_eligible")


def _contradictions(**registers) -> tuple[str, ...]:
    """Contradictions with every builder-owned and take-up register empty."""
    return us_register_contradictions(
        degenerate_reviewed_exclusions=(),
        documented_absent_inputs=(),
        seeded_variables=(),
        count_calibrated_variables=(),
        **registers,
    )


def _parity_lines(contradictions: tuple[str, ...]) -> list[str]:
    return [line for line in contradictions if _PARITY_GAP_PHRASE in line]


def test_default_parity_register_meets_the_signal_registers() -> None:
    # Every shipped parity gap demanded as signal must be caught through the
    # DEFAULT register path the build preflight uses.
    names = [gap.variable for gap in load_ecps_parity_known_gaps()]

    lines = _parity_lines(_contradictions(nonconstant_required_columns=names))

    assert sorted(line.split(":", 1)[0] for line in lines) == sorted(names)


def test_default_parity_gap_contradicts_a_seeded_take_up_column() -> None:
    # takes_up_dc_ptc is a shipped parity gap; seeding it nonconstant while the
    # register excuses it is the #377 pincer the preflight exists to catch.
    contradictions = us_register_contradictions(
        degenerate_reviewed_exclusions=(),
        documented_absent_inputs=(),
        seeded_variables=("takes_up_dc_ptc",),
        count_calibrated_variables=(),
    )

    assert [line.split(":", 1)[0] for line in _parity_lines(contradictions)] == [
        "takes_up_dc_ptc"
    ]


@pytest.mark.parametrize("spelling", _WIC_SPELLINGS)
def test_parity_gap_meets_the_live_column_under_either_spelling(spelling) -> None:
    # takes_up_wic_if_eligible is a coverage-manifest required column; a WIC
    # parity gap contradicts it whichever spelling the register uses.
    lines = _parity_lines(_contradictions(parity_known_gaps=(spelling,)))

    assert len(lines) == 1
    assert lines[0].startswith("takes_up_wic_if_eligible: required to carry signal")
    assert ("under its historical name 'would_claim_wic'" in lines[0]) is (
        spelling == "would_claim_wic"
    )


def test_parity_gap_named_under_both_spellings_is_refused() -> None:
    with pytest.raises(ValueError, match="would merge two exemptions"):
        _contradictions(parity_known_gaps=_WIC_SPELLINGS)


def test_gate_reports_a_parity_contradiction_as_a_failure() -> None:
    gate = us_register_consistency_gate(
        degenerate_reviewed_exclusions=(),
        documented_absent_inputs=(),
        seeded_variables=(),
        count_calibrated_variables=(),
        parity_known_gaps=("would_claim_wic",),
    )

    assert not gate.passed
    assert gate.details == {"contradictions": 1}
    assert gate.failures[0].startswith("takes_up_wic_if_eligible:")
