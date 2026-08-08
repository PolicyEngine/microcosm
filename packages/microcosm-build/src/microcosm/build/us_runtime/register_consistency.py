"""Cross-register consistency: no column may be required-to-signal AND excused.

Every US release register has a local anti-rot rule (a stale degenerate
exclusion fails, a stale parity gap fails, a stale coverage exclusion fails),
but nothing checked the registers against EACH OTHER. microcosm #377 is the
proof: main simultaneously seeded ``takes_up_tanf_if_eligible`` nonconstant
(take-up stage, #312/#315) and excused it as constant-at-default
(``US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS``). The degenerate-input gate's
cannot-rot rule then correctly failed every healthy seeded build — twice on
live runs (Build G 2026-07-07, Build I 2026-07-08) — because the two registers
demanded contradictory states of the same column. No build can pass both sides
of such a pincer:

- constant at the engine default -> the signal-side gate fails;
- carrying signal -> the excused-side cannot-rot rule fails.

This module makes the contradiction itself a first-class gate. The
**signal side** is every register that requires a column to carry non-default
signal:

- seeded take-up programs (``take_up_contract.json`` ``populace_treatment ==
  "seed"``; the take-up signal gate hard-fails a constant column);
- count-calibrated take-up programs (the export re-check hard-fails a
  count-calibrated column shipping at the engine default);
- the health-input nonconstant columns (the ACA source-stage gate);
- coverage-manifest ``required`` columns (the #369 input-coverage gate fails a
  required column that is absent or degenerate).

The **excused side** is every register that documents a column as legitimately
absent or constant-at-default:

- ``US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS`` (builder register, #286);
- coverage-manifest ``reviewed_exclusion`` columns (#369);
- eCPS parity known gaps (``ecps_parity_known_gaps.json``);
- ``US_DOCUMENTED_ABSENT_INPUTS`` (builder register, #351/#249).

Any column in a signal-side register and an excused-side register at once is a
contradiction that will abort a release run after the expensive source stages;
:func:`us_register_consistency_gate` fails it in CI and in a cheap build
preflight instead (microcosm #377 acceptance criteria).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.parity_reference import load_ecps_parity_known_gaps
from microcosm.build.us_runtime.release_input_coverage import (
    ReleaseInputCoverageManifest,
    load_release_input_coverage_manifest,
)
from microcosm.build.us_runtime.take_up_contract import (
    count_calibrated_take_up_programs,
    seeded_take_up_programs,
)

__all__ = [
    "us_register_consistency_gate",
    "us_register_contradictions",
]

#: Signal-side register labels, in report order. Each names the register AND
#: the gate that enforces it, so a failure line reads as the exact pincer.
_SEEDED_LABEL = (
    "seeded take-up programs (take_up_contract.json populace_treatment='seed'; "
    "the take-up signal gate fails a constant column)"
)
_COUNT_CALIBRATED_LABEL = (
    "count-calibrated take-up programs (take_up_contract.json "
    "populace_treatment='count_calibrated'; the export re-check fails an "
    "engine-default column)"
)
_NONCONSTANT_LABEL = (
    "nonconstant-required source-stage columns (the source-stage signal gates "
    "fail a constant column)"
)
_COVERAGE_REQUIRED_LABEL = (
    "coverage-manifest required columns (release_input_coverage_manifest.json "
    "status='required'; the input-coverage gate fails an absent or degenerate "
    "column)"
)

_DEGENERATE_LABEL = (
    "degenerate-input reviewed exclusions "
    "(US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS; the degenerate-input gate fails "
    "an excluded column that carries signal)"
)
_COVERAGE_EXCLUDED_LABEL = (
    "coverage-manifest reviewed exclusions "
    "(release_input_coverage_manifest.json status='reviewed_exclusion'; the "
    "input-coverage gate fails an excluded column that carries signal)"
)
_PARITY_GAP_LABEL = (
    "eCPS parity known gaps (ecps_parity_known_gaps.json; the parity gate "
    "fails a gap the candidate now populates)"
)
_DOCUMENTED_ABSENT_LABEL = (
    "documented-absent inputs (US_DOCUMENTED_ABSENT_INPUTS; the register "
    "asserts the column is not persisted at all)"
)


def us_register_contradictions(
    *,
    degenerate_reviewed_exclusions: Mapping[str, str] | Iterable[str],
    documented_absent_inputs: Mapping[str, str] | Iterable[str],
    nonconstant_required_columns: Iterable[str] = (),
    seeded_variables: Iterable[str] | None = None,
    count_calibrated_variables: Iterable[str] | None = None,
    coverage_manifest: ReleaseInputCoverageManifest | None = None,
    parity_known_gaps: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Every cross-register contradiction, one human-readable line each.

    A contradiction is a column present in a signal-side register (must carry
    non-default signal) and an excused-side register (documented as absent or
    constant-at-default) at once. The builder-owned registers
    (``US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS``,
    ``US_DOCUMENTED_ABSENT_INPUTS``, the health nonconstant tuple) are passed
    in; the packaged registers (take-up contract, coverage manifest, parity
    known gaps) default to the shipped resources.

    Returns:
        One line per (column, signal register, excused register) triple, in
        deterministic order. Empty means the registers are consistent.
    """
    if seeded_variables is None:
        seeded_variables = tuple(
            program.variable for program in seeded_take_up_programs()
        )
    if count_calibrated_variables is None:
        count_calibrated_variables = tuple(
            program.variable for program in count_calibrated_take_up_programs()
        )
    manifest = coverage_manifest or load_release_input_coverage_manifest()
    if parity_known_gaps is None:
        parity_known_gaps = tuple(load_ecps_parity_known_gaps())

    signal_registers: tuple[tuple[str, frozenset[str]], ...] = (
        (_SEEDED_LABEL, frozenset(seeded_variables)),
        (_COUNT_CALIBRATED_LABEL, frozenset(count_calibrated_variables)),
        (_NONCONSTANT_LABEL, frozenset(nonconstant_required_columns)),
        (_COVERAGE_REQUIRED_LABEL, frozenset(manifest.required_columns)),
    )
    excused_registers: tuple[tuple[str, frozenset[str]], ...] = (
        (_DEGENERATE_LABEL, frozenset(degenerate_reviewed_exclusions)),
        (_COVERAGE_EXCLUDED_LABEL, frozenset(manifest.reviewed_exclusions)),
        (_PARITY_GAP_LABEL, frozenset(parity_known_gaps)),
        (_DOCUMENTED_ABSENT_LABEL, frozenset(documented_absent_inputs)),
    )

    contradictions: list[str] = []
    for signal_label, signal_columns in signal_registers:
        for excused_label, excused_columns in excused_registers:
            if (
                signal_label is _COVERAGE_REQUIRED_LABEL
                and excused_label is _COVERAGE_EXCLUDED_LABEL
            ):
                # Structurally exclusive: one manifest column has one status.
                continue
            for column in sorted(signal_columns & excused_columns):
                contradictions.append(
                    f"{column}: required to carry signal by {signal_label} but "
                    f"excused as absent/degenerate by {excused_label}. No build "
                    "can pass both gates; remove the column from one register."
                )
    return tuple(contradictions)


def us_register_consistency_gate(
    *,
    degenerate_reviewed_exclusions: Mapping[str, str] | Iterable[str],
    documented_absent_inputs: Mapping[str, str] | Iterable[str],
    nonconstant_required_columns: Iterable[str] = (),
    seeded_variables: Iterable[str] | None = None,
    count_calibrated_variables: Iterable[str] | None = None,
    coverage_manifest: ReleaseInputCoverageManifest | None = None,
    parity_known_gaps: Iterable[str] | None = None,
    name: str = "us_register_consistency",
) -> GateResult:
    """The cross-register consistency verdict as a release gate.

    Passes iff :func:`us_register_contradictions` finds nothing. Run as a
    build preflight (before the expensive source stages, so a contradiction
    aborts in seconds, not after hours of staging — the Build G/Build I
    failure mode of #377) and asserted over the shipped registers in CI.
    """
    contradictions = us_register_contradictions(
        degenerate_reviewed_exclusions=degenerate_reviewed_exclusions,
        documented_absent_inputs=documented_absent_inputs,
        nonconstant_required_columns=nonconstant_required_columns,
        seeded_variables=seeded_variables,
        count_calibrated_variables=count_calibrated_variables,
        coverage_manifest=coverage_manifest,
        parity_known_gaps=parity_known_gaps,
    )
    return GateResult(
        name=name,
        passed=not contradictions,
        failures=contradictions,
        details={"contradictions": len(contradictions)},
    )
