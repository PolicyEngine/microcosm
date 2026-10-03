"""Property tests for the US immigration-status stage (microcosm #225, #767).

Every property below quotes the docstring, comment or runtime check in
``microcosm.build.us_runtime.immigration`` that promises it. Properties marked
*deduced* combine two stated mechanisms rather than restating one sentence
(for example, draws keyed by person identity plus threshold selection imply
row-order equivariance). The example tests live in ``test_us_immigration.py``.

Invariants checked, by function:

- ``_select_weight_to_target``: the selection is the candidates at or below
  one draw threshold, so tie groups are never split; a non-positive amount
  selects nothing; a feasible selection reaches the amount and overshoots it
  by less than its final tie group, and an infeasible one takes every
  candidate; it equals an independent group-walk reference; it is row-order
  equivariant (deduced).
- ``_stable_person_draws``: draws depend only on seed, salt and the most
  complete stable lineage (equal keys draw equally, different keys draw
  differently), reruns repeat and row order does not matter (deduced).
- ``_spill_pew_unauthorized_excess``: only in-scope residual non-citizens move,
  and only to EAD; Pew-universe membership never grows; a margin within the
  closure tolerance of its control spills nothing; every closable excess is
  closed and an unclosable one exhausts the removable pool; the undershoot is
  less than one flipped row; rows inside the other margin are touched only
  after every removable row outside it, and only when removing those leaves
  more than rounding residue (explicit example: the minimal exact-closure
  case, where 0.1 + 0.2 - 0.3 > 0 in binary floating point).
- The stage (``derive_us_immigration_status_from_manifest``): ``NONE`` pairs
  exactly with ``UNDOCUMENTED`` and ``CITIZEN`` with measured citizenship;
  source columns pass through; the status column is the documented function of
  SSN code, humanitarian mark and evidence; the indicator-documented pool is
  exactly the non-citizens carrying a documented indicator; humanitarian
  labels respect their exclusions and SSN pools, are mutually exclusive, reach
  their targets or exhaust their pools, precede the EAD spill, and sit inside
  the ASEC arrival windows the module's comments state; the output is
  bit-reproducible, row-order equivariant (deduced) and consistent across
  support clones; the Pew margins close whenever the residual pool can close
  them.
- ``with_us_immigration_inputs``: labels every person exactly as the manifest
  handler does, leaves weights and other tables alone, and is idempotent.
- Differential: stage output satisfies every structural check of
  ``us_immigration_composition_gate`` at the stage's own period (explicit
  example: a row whose DACA-cohort membership changes between 2024 and
  2025), and the spill's ``_pew_unauthorized_projection_mask`` selects
  exactly the rows the gate counts as Pew-unauthorized.
- ``reconcile_us_immigration_humanitarian_transfer``: its output satisfies the
  gate's structural contract; each receipt's achieved population is its
  immutable plus selected population; and
  ``us_immigration_humanitarian_transfer_selection_masks`` replays the
  reconciliation's mutable-row selections exactly (differential).

Independence limits, stated so no test claims more than it shows:

- The candidate pools (origin x arrival window x SSN pool) of the draw-target
  and transfer properties come from the module's own
  ``_humanitarian_draw_candidates``; those properties check selection against
  target, not the windows. Only
  ``test_humanitarian_rows_sit_inside_the_asec_windows_the_comments_state``
  checks windows, against values transcribed from the module's comments.
  That is a comment-versus-table consistency check, not external validation.
- Reconciliation raises ``RuntimeError`` before returning a receipt that
  misses its tie-mass bounds or changes an immutable row, so asserting those
  on returned receipts only confirms the receipt records what was checked.
- ``_facts`` reuses the module's arrival-midpoint and Cuba/Haiti code tables;
  the DACA thresholds come from the module docstring.

Hypothesis reaches the engine-free CI job through ``uv sync --all-packages``
(microcosm-graph's dev group), so it is imported directly: a missing
dependency fails loudly rather than skipping these tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import event, example, given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.immigration as immigration
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

#: Unit properties are cheap; stage properties run the full derivation per
#: example. ``database=None`` keeps runs from saving examples into the
#: checkout.
_UNIT = settings(max_examples=100, deadline=None, database=None)
_STAGE = settings(max_examples=50, deadline=None, database=None)
_TRANSFER = settings(max_examples=30, deadline=None, database=None)
_FRAME = settings(max_examples=20, deadline=None, database=None)

_TIME_PERIOD = 2024
#: Stage periods. The DACA cohort's age-at-entry test moves with the period,
#: so the stage and the composition gate must agree at each of them.
_PERIODS = (2024, 2025, 2030)

_HUMANITARIAN_STATUSES = frozenset(
    immigration._HUMANITARIAN_STATUS_BY_CATEGORY.values()
)
_INDICATOR_POOL_STATUSES = frozenset({"REFUGEE", "ASYLEE", "DEPORTATION_WITHHELD"})

#: Every origin in the humanitarian codebook tables.
_HUMANITARIAN_ORIGIN_CODES: tuple[int, ...] = tuple(
    sorted(
        {
            *(
                code
                for codes in immigration._PAROLE_ORIGIN_CODES.values()
                for code in codes
            ),
            *immigration._REFUGEE_ORIGIN_CODES,
            *immigration._ASYLEE_ORIGIN_CODES,
            *(
                code
                for codes, _ in immigration._TPS_ORIGIN_CODES.values()
                for code in codes
            ),
        }
    )
)
#: Birth countries the stage distinguishes: the United States, a generic
#: origin, Cuba/Haiti, and the humanitarian origins.
_BIRTH_CODES: tuple[int, ...] = tuple(
    sorted(
        {57, 120, 303, *immigration._CUBAN_HAITIAN_BIRTH_CODES}
        | set(_HUMANITARIAN_ORIGIN_CODES)
    )
)

#: ASEC-UA legal-status indicators as the module docstring lists them
#: ("Medicare/Medicaid/SSI/Social Security receipt, federal pensions,
#: IHS/CHAMPVA/military coverage, government employment, subsidized housing,
#: veteran status"), with the neutral value each column takes otherwise.
#: Pre-1982 arrival is drawn through ``PEINUSYR``. Naturalization-eligibility
#: is omitted: the code reads it from ``PRCITSHP == 4``, a citizen code, so it
#: can never move a ``PRCITSHP == 5`` row and no property here depends on it.
_NEUTRAL_INDICATORS: Mapping[str, float | int] = {
    "MCARE": 2,
    "CAID": 2,
    "IHSFLG": 2,
    "CHAMPVA": 2,
    "MIL": 2,
    "PEN_SC1": 0,
    "PEN_SC2": 0,
    "RESNSS1": 0,
    "RESNSS2": 0,
    "SS_YN": 2,
    "SSI_YN": 2,
    "PEIO1COW": 0,
    "A_MJOCC": 0,
    "PEAFEVER": 2,
    "SPM_CAPHOUSESUB": 0.0,
}
_INDICATORS: tuple[tuple[str, float | int], ...] = (
    ("MCARE", 1),
    ("CAID", 1),
    ("IHSFLG", 1),
    ("CHAMPVA", 1),
    ("MIL", 1),
    ("PEN_SC1", 3),
    ("PEN_SC2", 3),
    ("RESNSS1", 2),
    ("RESNSS2", 2),
    ("SS_YN", 1),
    ("SSI_YN", 1),
    ("PEIO1COW", 1),
    ("PEIO1COW", 2),
    ("PEIO1COW", 3),
    ("A_MJOCC", 11),
    ("PEAFEVER", 1),
    ("SPM_CAPHOUSESUB", 250.0),
)

#: ASEC arrival windows transcribed from the module's comments, not read from
#: its tables. Birth codes are the module's own (its comment says they were
#: "verified against Census CPS technical documentation"); the windows are the
#: comments' words. ``_RECENT_ARRIVAL_CODES``: "The 2024 ASEC distinguishes
#: 2020-2021 (27) from 2022-2024 (28) ... U4U and CHNV cannot draw from code
#: 27, while Operation Allies Welcome can"; ``_PAROLE_ORIGIN_CODES``: "Operation
#: Allies Welcome (Afghanistan), Uniting for Ukraine, and the
#: non-Cuban/Haitian CHNV nationalities".
_PAROLE_DOCUMENTED_WINDOWS: Mapping[int, frozenset[int]] = {
    200: frozenset({27, 28}),  # Afghanistan (Operation Allies Welcome)
    164: frozenset({28}),  # Ukraine (Uniting for Ukraine)
    315: frozenset({28}),  # Nicaragua (CHNV)
    373: frozenset({28}),  # Venezuela (CHNV)
}
#: ``_TPS_ORIGIN_CODES``: "Legacy designations bind hard (El Salvador
#: 2001-02-13 -> code 17; Honduras/Nicaragua 1998-12-30 -> code 16; Nepal
#: 2015-06-24 -> code 24)".
_TPS_DOCUMENTED_MAX_ARRIVAL_CODE: Mapping[int, int] = {
    312: 17,  # El Salvador
    314: 16,  # Honduras
    315: 16,  # Nicaragua
    229: 24,  # Nepal
}
#: ``_TPS_ORIGIN_CODES``: "Ukraine and Afghanistan TPS registrants are carried
#: by the parole draw ...; Haiti TPS is carried by CUBAN_HAITIAN_ENTRANT", and
#: Cuba/Haiti-born persons "are excluded from every humanitarian draw".
_TPS_DOCUMENTED_EXCLUDED_BIRTHS = frozenset({164, 200, 327, 332})
#: ``_ASYLEE_ARRIVAL_CODES``: "grantees in 2022-2024 overwhelmingly arrived
#: 2016+"; the 2016-2017 bin (midpoint 2017) is code 25.
_ASYLEE_DOCUMENTED_MIN_ARRIVAL_CODE = 25


def _draw_labels() -> tuple[str, ...]:
    labels: list[str] = []
    for category in immigration.HUMANITARIAN_STATUS_CATEGORIES:
        origins = immigration._PER_ORIGIN_HUMANITARIAN_CATEGORIES.get(category)
        if origins is None:
            labels.append(category)
        else:
            labels.extend(f"{category}:{origin}" for origin in origins)
    return tuple(labels)


_DRAW_LABELS = _draw_labels()

#: One zero-target draw per manifest label, for evaluating candidate pools.
_TEMPLATE_DRAWS: tuple[immigration.HumanitarianDraw, ...] = tuple(
    immigration.HumanitarianDraw(
        category=label.split(":", 1)[0],
        origin=label.split(":", 1)[1] if ":" in label else None,
        status=immigration._HUMANITARIAN_STATUS_BY_CATEGORY[label.split(":", 1)[0]],
        target=0.0,
        source="https://example.org/template",
    )
    for label in _DRAW_LABELS
)


def _tolerance(weights: np.ndarray) -> float:
    return 1e-9 * max(1.0, float(np.abs(weights).sum()))


# ---------------------------------------------------------------------------
# _select_weight_to_target
# ---------------------------------------------------------------------------

#: A few repeated draw values force the tie groups clones produce.
_TIED_DRAWS = (0.0, 0.125, 0.25, 0.5, 0.75, 0.875)


@st.composite
def _selection_cases(draw, *, integer_weights: bool = False):
    n = draw(st.integers(0, 24))
    weight = (
        st.integers(0, 50).map(float)
        if integer_weights
        else st.floats(0.0, 1e6, allow_nan=False, allow_subnormal=False)
    )
    weights = np.asarray(
        draw(st.lists(weight, min_size=n, max_size=n)), dtype=np.float64
    )
    draws = np.asarray(
        draw(
            st.lists(
                st.one_of(
                    st.sampled_from(_TIED_DRAWS),
                    st.floats(0.0, 1.0, exclude_max=True),
                ),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=np.float64,
    )
    candidates = np.asarray(
        draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool
    )
    total = float(weights[candidates].sum())
    amount = draw(
        st.one_of(
            st.floats(-10.0, 0.0),
            st.floats(0.0, max(1.5 * total, 1.0)),
            st.just(total),
        )
    )
    return candidates, weights, draws, amount


def _reference_selection(
    candidates: np.ndarray, weights: np.ndarray, draws: np.ndarray, amount: float
) -> np.ndarray:
    """Walk distinct draw values upward; stop at the first group reaching it."""

    if amount <= 0 or not candidates.any():
        return np.zeros(len(candidates), dtype=bool)
    cumulative = 0.0
    for value in sorted(set(draws[candidates].tolist())):
        cumulative += float(weights[candidates & (draws == value)].sum())
        if cumulative >= amount:
            return candidates & (draws <= value)
    return candidates.copy()


@_UNIT
@given(_selection_cases())
def test_selection_is_a_draw_threshold_prefix_that_never_splits_ties(case) -> None:
    """``_select_weight_to_target``: "every candidate at or below the draw
    where the cumulative weight first reaches ``amount`` is selected. Records
    sharing a draw ... land on the same side of the threshold" and "A
    non-positive ``amount`` selects nothing"."""

    candidates, weights, draws, amount = case
    selected = immigration._select_weight_to_target(candidates, weights, draws, amount)
    assert selected.dtype == bool and selected.shape == candidates.shape
    assert not (selected & ~candidates).any()
    if amount <= 0 or not candidates.any():
        assert not selected.any()
        return
    threshold = draws[selected].max()
    np.testing.assert_array_equal(selected, candidates & (draws <= threshold))


@_UNIT
@given(_selection_cases())
def test_selection_reaches_amount_and_overshoots_by_less_than_its_tie_group(
    case,
) -> None:
    """``_select_weight_to_target`` selects "~``amount`` of weighted mass":
    the cumulative weight "first reaches ``amount``" at the threshold and "the
    final tie group may overshoot the amount". When the candidates cannot
    reach it, every candidate is selected."""

    candidates, weights, draws, amount = case
    selected = immigration._select_weight_to_target(candidates, weights, draws, amount)
    if amount <= 0 or not candidates.any():
        return
    tolerance = _tolerance(weights)
    total = float(weights[candidates].sum())
    chosen = float(weights[selected].sum())
    if total < amount - tolerance:
        event("amount infeasible")
        np.testing.assert_array_equal(selected, candidates)
        return
    event("amount feasible")
    tie = selected & (draws == draws[selected].max())
    assert chosen >= amount - tolerance
    assert chosen - float(weights[tie].sum()) < amount + tolerance


@_UNIT
@given(_selection_cases(integer_weights=True))
def test_selection_matches_an_independent_group_walk(case) -> None:
    """Differential: the ``np.searchsorted`` implementation equals a plain
    group walk over distinct draw values (integer weights keep both sums
    exact, so the comparison is bitwise)."""

    candidates, weights, draws, amount = case
    np.testing.assert_array_equal(
        immigration._select_weight_to_target(candidates, weights, draws, amount),
        _reference_selection(candidates, weights, draws, amount),
    )


@_UNIT
@given(_selection_cases(integer_weights=True), st.data())
def test_selection_is_row_order_equivariant(case, data) -> None:
    """Deduced: a threshold on draws with whole tie groups on one side depends
    only on the (draw, weight) multiset, not on row positions. Integer weights
    keep cumulative sums order-independent."""

    candidates, weights, draws, amount = case
    order = np.asarray(
        data.draw(st.permutations(range(len(candidates)))), dtype=np.int64
    )
    selected = immigration._select_weight_to_target(candidates, weights, draws, amount)
    permuted = immigration._select_weight_to_target(
        candidates[order], weights[order], draws[order], amount
    )
    np.testing.assert_array_equal(permuted, selected[order])


# ---------------------------------------------------------------------------
# _stable_person_draws
# ---------------------------------------------------------------------------


@_UNIT
@given(
    keys=st.lists(st.integers(0, 6), min_size=1, max_size=16),
    seed=st.integers(0, 2**32),
    salt=st.text(max_size=8),
    data=st.data(),
)
def test_person_draws_are_keyed_by_identity_alone(keys, seed, salt, data) -> None:
    """``_stable_person_draws``: "Deterministic uniform draws keyed by stable
    person identity" that give "every clone of one source person the same
    draw". Row-order equivariance is deduced from the per-key hash."""

    person = pd.DataFrame({"person_id": keys})
    draws = immigration._stable_person_draws(person, seed=seed, salt=salt)
    assert draws.shape == (len(keys),)
    assert ((draws >= 0.0) & (draws < 1.0)).all()
    key_array = np.asarray(keys)
    for key in set(keys):
        assert len(set(draws[key_array == key].tolist())) == 1
    np.testing.assert_array_equal(
        draws, immigration._stable_person_draws(person, seed=seed, salt=salt)
    )
    order = list(data.draw(st.permutations(range(len(keys)))))
    reordered = person.iloc[order].reset_index(drop=True)
    np.testing.assert_array_equal(
        immigration._stable_person_draws(reordered, seed=seed, salt=salt),
        draws[order],
    )


def _equal_draws_iff_equal_keys(draws: np.ndarray, keys: list[tuple]) -> None:
    """Draws agree exactly for equal keys and differ for different ones (a
    64-bit blake2b collision between distinct keys has probability ~2**-64)."""

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert (draws[i] == draws[j]) == (keys[i] == keys[j]), (i, j)


@st.composite
def _lineage_rows(draw) -> list[tuple[int | None, int | None]]:
    """(source_household_id, source_person_id) rows, each lineage alternative
    about equally often: complete, household missing somewhere (legacy pair),
    or source person missing somewhere (``person_id`` fallback). Small id
    ranges make one source person recur across households."""

    n = draw(st.integers(2, 10))
    ids = st.integers(1, 3)
    rows = draw(st.lists(st.tuples(ids, ids), min_size=n, max_size=n))
    mode = draw(st.sampled_from(("full", "legacy", "person_id")))
    if mode == "full":
        return rows
    gaps = draw(st.sets(st.integers(0, n - 1), min_size=1))
    if mode == "legacy":
        return [(None, p) if i in gaps else (h, p) for i, (h, p) in enumerate(rows)]
    household_gaps = draw(st.booleans())
    return [
        ((None if household_gaps else h), None) if i in gaps else (h, p)
        for i, (h, p) in enumerate(rows)
    ]


@_UNIT
@given(rows=_lineage_rows(), seed=st.integers(0, 2**32))
def test_person_draws_use_the_most_complete_lineage(rows, seed) -> None:
    """``_stable_person_draws``: support clones key on ``source_year`` /
    ``source_household_id`` / ``source_person_id``, and "frames without full
    source ids fall back to the legacy source pair or ``person_id``"; with no
    complete alternative it raises. Draws are equal exactly when the chosen
    lineage is equal, so two rows sharing ``source_year`` and
    ``source_person_id`` but not ``source_household_id`` draw differently
    whenever the full lineage is complete (a draw that ignored the household
    would tie them)."""

    household = [household for household, _ in rows]
    source_person = [person for _, person in rows]
    person = pd.DataFrame(
        {
            "source_year": [2024] * len(rows),
            "source_household_id": pd.array(household, dtype="Int64"),
            "source_person_id": pd.array(source_person, dtype="Int64"),
            "person_id": np.arange(1, len(rows) + 1),
        }
    )

    def draws(frame: pd.DataFrame) -> np.ndarray:
        return immigration._stable_person_draws(frame, seed=seed, salt="lineage")

    full = draws(person)
    if None not in source_person and None not in household:
        event("full lineage complete")
        if len(set(source_person)) < len(
            set(zip(household, source_person, strict=True))
        ):
            event("same source person in different households")
        np.testing.assert_array_equal(full, draws(person.drop(columns="person_id")))
        _equal_draws_iff_equal_keys(
            full, list(zip(household, source_person, strict=True))
        )
    elif None not in source_person:
        event("legacy lineage")
        np.testing.assert_array_equal(
            full, draws(person.drop(columns="source_household_id"))
        )
        _equal_draws_iff_equal_keys(full, [(key,) for key in source_person])
    else:
        event("person_id fallback")
        np.testing.assert_array_equal(full, draws(person[["person_id"]]))
        _equal_draws_iff_equal_keys(full, [(key,) for key in range(len(rows))])
        with pytest.raises(SourceRuntimeError, match="lineage"):
            draws(person.drop(columns="person_id"))


# ---------------------------------------------------------------------------
# _spill_pew_unauthorized_excess
# ---------------------------------------------------------------------------

_MARKS = ("", "", *sorted(_HUMANITARIAN_STATUSES))
#: Decimal weights whose binary sums leave rounding residue (0.1 + 0.2 is not
#: 0.3), the case the spill's closure tolerance exists for.
_DECIMAL_WEIGHTS = (0.1, 0.2, 0.3, 0.7, 1.1)
_WEIGHT = st.one_of(
    st.just(0.0),
    st.integers(1, 5_000).map(float),
    st.sampled_from(_DECIMAL_WEIGHTS),
    st.floats(0.5, 5e5, allow_nan=False, allow_subnormal=False),
)
#: Spill rows lean further toward decimal weights: the exact-closure residue
#: needs them on both sides of the tier boundary.
_SPILL_WEIGHT = st.one_of(st.sampled_from(_DECIMAL_WEIGHTS), _WEIGHT)
#: ``_spill_pew_unauthorized_excess``: "Treat anything within a billionth of
#: the in-scope weight as closed".
_DOCUMENTED_CLOSURE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _SpillCase:
    ssn: np.ndarray
    marks: np.ndarray
    weights: np.ndarray
    noncitizens: np.ndarray
    scope: np.ndarray
    preserve: np.ndarray
    retained: np.ndarray
    target: float
    seed: int


def _projection_of(
    ssn: np.ndarray, marks: np.ndarray, retained: np.ndarray
) -> np.ndarray:
    return immigration._pew_unauthorized_projection_mask(
        ssn_codes=ssn,
        humanitarian_marks=marks,
        retained_ead_cohort=retained,
    )


@st.composite
def _spill_cases(draw) -> _SpillCase:
    """Mostly residual in-scope non-citizens, with a target that is a share of
    the in-scope Pew count (so the spill has work to do), exactly that count
    less a subset of first-tier removable rows (the exact-closure case), or at
    or above the count (so it must do nothing)."""

    n = draw(st.integers(2, 24), label="rows")
    rows = draw(
        st.lists(
            st.tuples(
                st.sampled_from((True, True, True, False)),
                st.sampled_from((0, 0, 0, 2, 3)),
                st.sampled_from(_MARKS),
                st.sampled_from((True, True, False)),
                st.booleans(),
                st.sampled_from((False, False, False, True)),
                _SPILL_WEIGHT,
            ),
            min_size=n,
            max_size=n,
        )
    )
    noncitizens = np.asarray([row[0] for row in rows], dtype=bool)
    ssn = np.where(noncitizens, [row[1] for row in rows], 1).astype(np.int64)
    # The stage never marks a residual (code 0) row: parole/TPS selections
    # leave that pool when they are marked.
    marks = np.where(noncitizens & (ssn != 0), [row[2] for row in rows], "").astype(
        "U24"
    )
    scope = np.asarray([row[3] for row in rows], dtype=bool)
    preserve = np.asarray([row[4] for row in rows], dtype=bool)
    retained = np.asarray([row[5] for row in rows], dtype=bool)
    weights = np.asarray([row[6] for row in rows], dtype=np.float64)
    initial = float(weights[scope & _projection_of(ssn, marks, retained)].sum())
    kind = draw(
        st.sampled_from(("share", "exact", "exact", "above")), label="target kind"
    )
    if kind == "share":
        target = draw(st.floats(0.0, 1.0)) * initial
    elif kind == "exact":
        # Removing the whole first tier closes the gap exactly in real
        # arithmetic, so the first tier's passes must take all of it; with
        # decimal weights the recomputed sum then differs from the target by
        # rounding residue alone.
        first_tier = scope & ~preserve & noncitizens & (ssn == 0) & ~retained
        subset = first_tier
        if not draw(st.booleans(), label="whole first tier"):
            subset = first_tier & np.asarray(
                draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool
            )
        target = max(0.0, initial - float(weights[subset].sum()))
    else:
        target = initial * draw(st.floats(1.0, 1.5))
    event(f"spill target: {kind}")
    return _SpillCase(
        ssn=ssn,
        marks=marks,
        weights=weights,
        noncitizens=noncitizens,
        scope=scope,
        preserve=preserve,
        retained=retained,
        target=target,
        seed=draw(st.integers(0, 2**31 - 1)),
    )


def _projection(case: _SpillCase, ssn: np.ndarray) -> np.ndarray:
    return _projection_of(ssn, case.marks, case.retained)


def _spill(case: _SpillCase) -> tuple[np.ndarray, np.ndarray]:
    """Run the spill on copies; return (codes after, marks after)."""

    ssn = case.ssn.copy()
    marks = case.marks.copy()
    immigration._spill_pew_unauthorized_excess(
        pd.DataFrame({"person_id": np.arange(1, len(ssn) + 1)}),
        ssn,
        marks,
        case.weights.copy(),
        noncitizens=case.noncitizens,
        scope=case.scope,
        preserve_scope=case.preserve,
        retained_ead_cohort=case.retained,
        target=case.target,
        seed=case.seed,
        salt="immigration:property",
    )
    return ssn, marks


@_UNIT
@given(_spill_cases())
def test_spill_moves_only_in_scope_residual_noncitizens_and_only_to_ead(
    case,
) -> None:
    """``_spill_pew_unauthorized_excess`` "Spill[s] a broad Pew-universe margin
    to EAD": candidates are ``(ssn_codes == 0) & noncitizens & scope`` and the
    only write is ``ssn_codes[selected] = 2``. It is deterministic."""

    after, marks = _spill(case)
    changed = after != case.ssn
    if changed.any():
        event("the spill moved rows")
    assert not (changed & ~((case.ssn == 0) & case.noncitizens & case.scope)).any()
    assert (after[changed] == 2).all()
    np.testing.assert_array_equal(marks, case.marks)
    np.testing.assert_array_equal(_spill(case)[0], after)


@_UNIT
@given(_spill_cases())
def test_spill_never_grows_the_pew_universe(case) -> None:
    """The spill exists to reduce a Pew-universe excess: moving a residual row
    to EAD either removes it from ``_pew_unauthorized_projection_mask`` or,
    for "DACA and residual Cuban/Haitian cohort rows", leaves it counted.
    Membership therefore only shrinks, in scope and everywhere else, so the
    weighted Pew count never rises."""

    after, _ = _spill(case)
    before_mask = _projection(case, case.ssn)
    after_mask = _projection(case, after)
    assert not (after_mask & ~before_mask).any()


@_UNIT
@given(_spill_cases())
def test_spill_leaves_a_margin_within_tolerance_of_its_control_untouched(
    case,
) -> None:
    """``_select_weight_to_target``: "a count already at or below its control
    spills nothing"; the spill returns while ``current_excess() <=
    closed_within``, and its comment says "Treat anything within a billionth
    of the in-scope weight as closed"."""

    included = _projection(case, case.ssn)
    excess = float(case.weights[case.scope & included].sum()) - case.target
    closed_within = _DOCUMENTED_CLOSURE_TOLERANCE * max(
        1.0, float(case.weights[case.scope].sum())
    )
    if excess > closed_within:
        return
    event("margin already closed")
    after, _ = _spill(case)
    np.testing.assert_array_equal(after, case.ssn)


@_UNIT
@given(_spill_cases())
def test_spill_closes_every_closable_excess_and_otherwise_exhausts_the_pool(
    case,
) -> None:
    """The workers/students controls are enforced "until the broad reported
    universes match their controls" (module docstring, step 3): each pass
    selects the current excess, and the corrective draw "spills rows outside
    those retained cohorts". So the in-scope Pew count ends at or below the
    target whenever removing every in-scope residual non-retained row could
    get it there, and otherwise every such row has been spilled."""

    tolerance = _tolerance(case.weights)
    after, _ = _spill(case)
    initial = float(case.weights[case.scope & _projection(case, case.ssn)].sum())
    final = float(case.weights[case.scope & _projection(case, after)].sum())
    removable = case.scope & case.noncitizens & (case.ssn == 0) & ~case.retained
    floor = initial - float(case.weights[removable].sum())
    if floor <= case.target - tolerance:
        event("excess closable")
        assert final <= case.target + tolerance
    elif floor > case.target + tolerance:
        event("excess not closable")
        assert (after[removable] == 2).all()


@_UNIT
@given(_spill_cases())
def test_spill_undershoots_its_control_by_less_than_one_spilled_row(case) -> None:
    """Each pass asks ``_select_weight_to_target`` for exactly the current
    excess, which "may overshoot the amount" only by its final tie group
    (one row here: every row has its own ``person_id`` draw). The in-scope
    Pew count therefore never lands a full spilled row below the target."""

    after, _ = _spill(case)
    changed = after != case.ssn
    if not changed.any():
        return
    final = float(case.weights[case.scope & _projection(case, after)].sum())
    largest = float(case.weights[changed].max())
    assert final > case.target - largest - _tolerance(case.weights)


#: The minimal exact-closure case: in real arithmetic the student control 0.3
#: equals the two worker-students' 0.1 + 0.2, so spilling the one non-worker
#: student (weight 1.0) closes the gap, yet ``0.1 + 0.2 - 0.3`` is 5.6e-17 in
#: binary floating point. ``test_us_immigration.py`` pins the same case end
#: to end through the stage.
_EXACT_CLOSURE_RESIDUE_CASE = _SpillCase(
    ssn=np.zeros(3, dtype=np.int64),
    marks=np.full(3, "", dtype="U24"),
    weights=np.asarray([1.0, 0.1, 0.2]),
    noncitizens=np.ones(3, dtype=bool),
    scope=np.ones(3, dtype=bool),
    preserve=np.asarray([False, True, True]),
    retained=np.zeros(3, dtype=bool),
    target=0.3,
    seed=0,
)


@_UNIT
@given(_spill_cases())
@example(_EXACT_CLOSURE_RESIDUE_CASE)
def test_spill_exhausts_rows_outside_the_other_margin_first(case) -> None:
    """``_spill_pew_unauthorized_excess``: "Rows outside the other controlled
    margin are exhausted before any overlapping row is touched" and "a row
    inside the other margin is spilled only once no row outside it can close
    the gap"; the closure tolerance ("anything within a billionth of the
    in-scope weight") exists so that rounding residue "never opens the next
    tier". If any row inside the preserved margin moved, every removable
    (non-retained residual) in-scope row outside it moved too, and removing
    all of them still left more than rounding residue above the target."""

    after, _ = _spill(case)
    changed = after != case.ssn
    if not (changed & case.preserve).any():
        return
    event("a row inside the other margin moved")
    outside = (
        case.scope
        & ~case.preserve
        & case.noncitizens
        & (case.ssn == 0)
        & ~case.retained
    )
    assert (after[outside] == 2).all()
    initial = float(case.weights[case.scope & _projection(case, case.ssn)].sum())
    closed_within = _DOCUMENTED_CLOSURE_TOLERANCE * max(
        1.0, float(case.weights[case.scope].sum())
    )
    left_after_outside = initial - float(case.weights[outside].sum()) - case.target
    assert left_after_outside > 0.5 * closed_within


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _generic_asec_row(weight):
    return st.fixed_dictionaries(
        {
            "PRCITSHP": st.sampled_from((1, 2, 3, 4, 5, 5, 5, 5, 5)),
            # Biased toward the origins and recent arrivals the humanitarian
            # draws need.
            "PENATVTY": st.one_of(
                st.sampled_from(_BIRTH_CODES),
                st.sampled_from(_HUMANITARIAN_ORIGIN_CODES),
            ),
            "PEINUSYR": st.one_of(
                st.integers(0, 28), st.integers(17, 28), st.integers(24, 28)
            ),
            "A_AGE": st.integers(0, 85),
            "A_MARITL": st.sampled_from((1, 2, 4, 7)),
            "A_SPOUSE": st.sampled_from((0, 1)),
            "A_HSCOL": st.sampled_from((0, 1, 2)),
            "A_LFSR": st.sampled_from(immigration._CPS_LABOR_FORCE_STATUS_DOMAIN),
            "indicator": st.one_of(st.none(), st.none(), st.sampled_from(_INDICATORS)),
            "person_weight": weight,
        }
    )


@st.composite
def _humanitarian_profile_row(draw, weight) -> dict:
    """A non-citizen shaped for one humanitarian draw or for the DACA cohort,
    or one arrival bin outside its window (so windows are exercised from both
    sides).

    Inputs only: the origin and arrival tables here come from the module so
    the candidate pools are reachable; no oracle reads them.
    """

    category = draw(
        st.sampled_from((*immigration.HUMANITARIAN_STATUS_CATEGORIES, "daca"))
    )
    age = draw(st.one_of(st.integers(25, 60), st.integers(0, 85)))
    if category == "daca":
        # Arrived by 2007 (module docstring, step 6) at an age measured at
        # 2024: under 16 (the cohort at every period drawn) or 16-21 (outside
        # it at 2024, inside it by 2030, so cohort membership depends on the
        # period). Sometimes Cuba/Haiti-born, where DACA and the entrant
        # class compete.
        birth = draw(st.sampled_from((303, 312, 327, 332)))
        arrival = draw(st.integers(8, 20))
        midpoint = immigration._ARRIVAL_YEAR_MIDPOINTS[arrival]
        entry_age_at_2024 = draw(st.one_of(st.integers(0, 15), st.integers(16, 21)))
        age = max(15, (_TIME_PERIOD - midpoint) + entry_age_at_2024)
    elif category == "paroled_one_year":
        origin = draw(st.sampled_from(tuple(immigration._PAROLE_ORIGIN_CODES)))
        birth = draw(st.sampled_from(immigration._PAROLE_ORIGIN_CODES[origin]))
        arrival = draw(st.sampled_from(immigration._PAROLE_ASEC_ARRIVAL_CODES[origin]))
    elif category == "refugee":
        birth = draw(st.sampled_from(immigration._REFUGEE_ORIGIN_CODES))
        arrival = 28
    elif category == "asylee":
        birth = draw(st.sampled_from(immigration._ASYLEE_ORIGIN_CODES))
        arrival = draw(st.sampled_from(immigration._ASYLEE_ARRIVAL_CODES))
    elif category == "deportation_withheld":
        birth = draw(st.sampled_from(_BIRTH_CODES))
        arrival = draw(st.integers(1, immigration._WITHHELD_MAX_ARRIVAL_CODE))
    else:
        codes, latest = draw(
            st.sampled_from(tuple(immigration._TPS_ORIGIN_CODES.values()))
        )
        birth = draw(st.sampled_from(codes))
        arrival = draw(st.integers(1, latest))
    if draw(st.integers(0, 4)) == 0:
        arrival = min(28, max(1, arrival + draw(st.sampled_from((-1, 1)))))
    return {
        "PRCITSHP": draw(st.sampled_from((5, 5, 5, 4))),
        "PENATVTY": birth,
        "PEINUSYR": arrival,
        "A_AGE": age,
        "A_MARITL": draw(st.sampled_from((1, 2, 4, 7))),
        "A_SPOUSE": draw(st.sampled_from((0, 1))),
        "A_HSCOL": draw(st.sampled_from((0, 1, 2))),
        "A_LFSR": draw(st.sampled_from(immigration._CPS_LABOR_FORCE_STATUS_DOMAIN)),
        # Refugee, asylee and withholding draws need an indicator; parole and
        # TPS draw from either pool; a DACA-cohort row needs the residual
        # pool to reach an EAD.
        "indicator": draw(
            st.none()
            if category == "daca"
            else st.one_of(
                st.none(), st.sampled_from(_INDICATORS), st.sampled_from(_INDICATORS)
            )
        ),
        "person_weight": draw(weight),
    }


def _asec_row(weight):
    return st.one_of(
        _generic_asec_row(weight),
        _generic_asec_row(weight),
        _humanitarian_profile_row(weight),
    )


def _asec_table(rows: list[dict]) -> pd.DataFrame:
    records = []
    for index, row in enumerate(rows):
        record = {**_NEUTRAL_INDICATORS, **row, "person_id": index + 1}
        indicator = record.pop("indicator")
        if indicator is not None:
            column, value = indicator
            record[column] = value
        records.append(record)
    return pd.DataFrame(records)


def _manifest_parameters(
    *,
    workers: float,
    students: float,
    anchor: float,
    targets: Mapping[str, float],
) -> dict[str, object]:
    """``derive_immigration_status`` parameters in the packaged manifest shape."""

    humanitarian: dict[str, object] = {}
    for category in immigration.HUMANITARIAN_STATUS_CATEGORIES:
        origins = immigration._PER_ORIGIN_HUMANITARIAN_CATEGORIES.get(category)
        if origins is None:
            humanitarian[category] = {
                "target": targets[category],
                "source": f"https://example.org/{category}",
            }
        else:
            humanitarian[category] = {
                origin: {
                    "target": targets[f"{category}:{origin}"],
                    "source": f"https://example.org/{category}/{origin}",
                }
                for origin in origins
            }
    return {
        "seed_from_build_config": True,
        "time_period_from_build_config": True,
        "undocumented_workers": {
            "target": workers,
            "source": "https://example.org/workers",
        },
        "undocumented_students": {
            "target": students,
            "source": "https://example.org/students",
        },
        "undocumented_population_anchor": {
            "value": anchor,
            "source": "https://example.org/population",
        },
        "humanitarian_status_stocks": humanitarian,
    }


@dataclass(frozen=True)
class _StageCase:
    person: pd.DataFrame
    workers: float
    students: float
    anchor: float
    targets: Mapping[str, float]
    seed: int
    time_period: int

    @property
    def parameters(self) -> dict[str, object]:
        return _manifest_parameters(
            workers=self.workers,
            students=self.students,
            anchor=self.anchor,
            targets=self.targets,
        )

    @property
    def controls(self) -> immigration.ImmigrationControls:
        return immigration._controls_from_parameters(self.parameters)


def _indicator_mask(person: pd.DataFrame) -> np.ndarray:
    """Rows carrying a documented indicator (see ``_INDICATORS``)."""

    indicator = np.isin(person["PEINUSYR"].to_numpy(), range(1, 8))
    for name, value in _INDICATORS:
        indicator |= person[name].to_numpy() == value
    return indicator


def _scopes(person: pd.DataFrame) -> dict[str, np.ndarray]:
    """The two controlled margins as step 3 defines them: workers "measured
    with ASEC ``A_LFSR`` at age 16+" and students."""

    return {
        "workers": (person["A_AGE"].to_numpy() >= 16)
        & np.isin(person["A_LFSR"].to_numpy(), (1, 2, 3, 4)),
        "students": person["A_HSCOL"].to_numpy() == 2,
    }


def _residual_masses(person: pd.DataFrame) -> dict[str, float]:
    """Each margin's pre-stage residual mass (indicator-free non-citizens).

    The stage's pre-spill Pew count in a margin is at least this: residual
    rows either stay residual or, if a parole/TPS draw takes them, keep a
    Pew-included mark. A control below it therefore makes the spill work.
    """

    residual = (person["PRCITSHP"].to_numpy() == 5) & ~_indicator_mask(person)
    weights = person["person_weight"].to_numpy(dtype=np.float64)
    return {
        name: float(weights[residual & scope].sum())
        for name, scope in _scopes(person).items()
    }


def _pool_masses(person: pd.DataFrame, time_period: int) -> dict[str, float]:
    """Each draw's weighted candidate pool before any draw runs.

    Scaling targets to these pools exercises both sides of every draw (a pool
    that suffices and one that runs short) instead of mostly saturating.
    """

    noncitizen = person["PRCITSHP"].to_numpy() == 5
    codes = np.where(noncitizen, np.where(_indicator_mask(person), 3, 0), 1)
    profile = immigration._source_aware_immigration_profile(
        person, time_period=time_period
    )
    weights = person["person_weight"].to_numpy(dtype=np.float64)
    return {
        template.label: float(
            weights[
                immigration._humanitarian_draw_candidates(
                    template, ssn_codes=codes, profile=profile
                )
            ].sum()
        )
        for template in _TEMPLATE_DRAWS
    }


@st.composite
def _target(draw, pool: float, scale: float) -> float:
    """Zero, a share of the draw's own pool, or occasionally a pool-blind stock."""

    if draw(st.integers(0, 9)) == 0:
        return draw(st.floats(1e-3, 0.1)) * scale
    return draw(st.one_of(st.just(0.0), st.floats(0.01, 1.5))) * pool


@st.composite
def _control(draw, mass: float, scale: float) -> float:
    """A worker/student control as a share of the margin's residual mass:
    below it (the spill must act), near it, or above it (it may not)."""

    share = draw(
        st.one_of(st.floats(0.01, 0.99), st.floats(0.99, 1.01), st.floats(1.01, 2.0))
    )
    return share * (mass if mass > 0 else scale)


@st.composite
def _stage_cases(
    draw,
    *,
    integer_weights: bool = False,
    min_size: int = 6,
    max_size: int = 32,
    time_periods: tuple[int, ...] = _PERIODS,
) -> _StageCase:
    weight = st.integers(0, 2_000).map(float) if integer_weights else _WEIGHT
    n = draw(st.integers(min_size, max_size), label="rows")
    rows = draw(st.lists(_asec_row(weight), min_size=n, max_size=n))
    person = _asec_table(rows)
    time_period = draw(st.sampled_from(time_periods), label="time period")
    scale = max(1.0, float(person["person_weight"].sum()))
    pools = _pool_masses(person, time_period)
    residual = _residual_masses(person)
    return _StageCase(
        person=person,
        workers=draw(_control(residual["workers"], scale), label="workers"),
        students=draw(_control(residual["students"], scale), label="students"),
        anchor=draw(st.floats(0.1, 2.0)) * scale,
        targets={label: draw(_target(pools[label], scale)) for label in _DRAW_LABELS},
        seed=draw(st.integers(0, 2**31 - 1)),
        time_period=time_period,
    )


def _derive(
    person: pd.DataFrame,
    parameters: Mapping[str, object],
    *,
    seed: int,
    time_period: int,
) -> pd.DataFrame:
    operation = SourceOperationSpec.from_mapping(
        {"kind": "derive_immigration_status", **parameters}
    )
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=seed, target_year=time_period),
        tables={},
    )
    return immigration.derive_us_immigration_status_from_manifest(
        person, operation, context
    )


def _run(case: _StageCase) -> pd.DataFrame:
    return _derive(
        case.person,
        case.parameters,
        seed=case.seed,
        time_period=case.time_period,
    )


@dataclass(frozen=True)
class _Facts:
    """Evidence recomputed from the raw columns and the documented rules."""

    weights: np.ndarray
    noncitizen: np.ndarray
    ssn: np.ndarray
    status: np.ndarray
    humanitarian: np.ndarray
    cuban_haitian_born: np.ndarray
    cuban_haitian_class: np.ndarray
    daca_cohort: np.ndarray
    indicator: np.ndarray
    worker: np.ndarray
    student: np.ndarray

    @property
    def retained(self) -> np.ndarray:
        """The spill's retained EAD cohort: DACA plus the Cuban/Haitian class
        outside it (module docstring, step 3: "Residual Cuban/Haitian rows
        that receive an EAD also stay in that universe")."""

        return self.daca_cohort | (self.cuban_haitian_class & ~self.daca_cohort)


def _facts(person: pd.DataFrame, output: pd.DataFrame, time_period: int) -> _Facts:
    def column(name: str) -> np.ndarray:
        return person[name].to_numpy()

    arrival_code = column("PEINUSYR").astype(np.int64)
    arrival_year = np.asarray(
        [
            immigration._ARRIVAL_YEAR_MIDPOINTS.get(int(code), time_period)
            for code in arrival_code
        ],
        dtype=np.int64,
    )
    age = column("A_AGE").astype(np.int64)
    age_at_entry = np.maximum(0, age - (time_period - arrival_year))
    # "DACA applies the statutory cohort test (arrived by 2007 before age 16,
    # aged 15+)" (module docstring, step 6).
    daca_cohort = (arrival_year <= 2007) & (age_at_entry < 16) & (age >= 15)
    cuban_haitian_born = np.isin(
        column("PENATVTY"), immigration._CUBAN_HAITIAN_BIRTH_CODES
    )
    status = output["immigration_status_str"].astype(str).to_numpy()
    scopes = _scopes(person)
    return _Facts(
        weights=person["person_weight"].to_numpy(dtype=np.float64),
        noncitizen=column("PRCITSHP") == 5,
        ssn=output["ssn_card_type"].astype(str).to_numpy(),
        status=status,
        humanitarian=np.isin(status, list(_HUMANITARIAN_STATUSES)),
        cuban_haitian_born=cuban_haitian_born,
        # "the nationality-plus-arrival class" (module docstring, step 6).
        cuban_haitian_class=cuban_haitian_born
        & (arrival_year >= immigration._CUBAN_HAITIAN_ARRIVAL_CUTOFF),
        daca_cohort=daca_cohort,
        indicator=_indicator_mask(person),
        worker=scopes["workers"],
        student=scopes["students"],
    )


def _frame(person: pd.DataFrame, weights: np.ndarray) -> Frame:
    """One person per household, weighted at the household."""

    person = person.drop(columns=["person_weight"], errors="ignore").copy()
    ids = np.arange(1, len(person) + 1, dtype=np.int64)
    person = person.reset_index(drop=True)
    for entity in US_SCHEMA.group_entities:
        person[US_SCHEMA.membership_column(entity)] = ids
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _codes(ssn: np.ndarray) -> np.ndarray:
    """Engine SSN names to the stage's integer codes, via the module's map."""

    return immigration._ssn_name_codes(pd.DataFrame({"ssn_card_type": ssn}))


def _pre_draw_codes(ssn: np.ndarray) -> np.ndarray:
    """SSN codes as the humanitarian draws saw them.

    Indicators set code 3 and nothing later changes it; the draws and the
    spill only move code 0 to code 2. So every final EAD row was residual when
    the draws ran.
    """

    return np.select(
        [ssn == "CITIZEN", ssn == "OTHER_NON_CITIZEN"], [1, 3], default=0
    ).astype(np.int64)


@_STAGE
@given(_stage_cases())
def test_stage_pairs_undocumented_with_no_ssn_and_citizens_with_citizenship(
    case,
) -> None:
    """Module docstring: "``UNDOCUMENTED`` enum remains exactly paired with
    ``ssn_card_type=NONE``"; "Citizenship is measured, not imputed.
    ``PRCITSHP`` in {1..4} → citizens"; "``CONDITIONAL_ENTRANT`` is
    deliberately not emitted"; and "The stage writes two PolicyEngine-US
    person input columns" (the source columns pass through unchanged)."""

    output = _run(case)
    assert list(output.columns) == [
        *case.person.columns,
        *immigration.US_IMMIGRATION_OUTPUT_COLUMNS,
    ]
    pd.testing.assert_frame_equal(output[list(case.person.columns)], case.person)
    ssn = output["ssn_card_type"].to_numpy()
    status = output["immigration_status_str"].to_numpy()
    assert set(ssn) <= set(immigration.SSN_CARD_TYPE_VALUES)
    assert set(status) <= set(immigration.IMMIGRATION_STATUS_VALUES)
    assert "CONDITIONAL_ENTRANT" not in set(status)
    np.testing.assert_array_equal(ssn == "NONE", status == "UNDOCUMENTED")
    citizen = np.isin(case.person["PRCITSHP"].to_numpy(), (1, 2, 3, 4))
    np.testing.assert_array_equal(ssn == "CITIZEN", citizen)
    np.testing.assert_array_equal(status == "CITIZEN", citizen)
    if (ssn == "NONE").any() and (ssn == "NON_CITIZEN_VALID_EAD").any():
        event("output carries both NONE and EAD rows")


@_STAGE
@given(_stage_cases())
def test_stage_status_is_the_documented_function_of_ssn_marks_and_evidence(
    case,
) -> None:
    """Reference implementation of step 6 and ``_derive_immigration_status``:
    citizens are ``CITIZEN``, ``NONE`` is ``UNDOCUMENTED``, a humanitarian
    mark wins, then "``DACA`` applies the statutory cohort test ... to EAD
    holders; ``CUBAN_HAITIAN_ENTRANT`` applies the nationality-plus-arrival
    class to documented non-citizens; every other documented non-citizen stays
    ``LEGAL_PERMANENT_RESIDENT``". DACA outranks the Cuban/Haitian class only
    for EAD holders (``_special_status_masks``: "a qualifying EAD holder in
    both cohorts retains DACA"), so an indicator-documented Cuba-born member
    of the DACA cohort is a Cuban/Haitian entrant. A mark never hides one of
    those tags: "Humanitarian draws exclude Cuba/Haiti-born and the DACA
    cohort, so the marks are disjoint from both tags"."""

    output = _run(case)
    facts = _facts(case.person, output, case.time_period)
    documented = np.isin(facts.ssn, ("NON_CITIZEN_VALID_EAD", "OTHER_NON_CITIZEN"))
    daca = (facts.ssn == "NON_CITIZEN_VALID_EAD") & facts.daca_cohort
    cuban_haitian = documented & facts.cuban_haitian_class & ~daca
    if daca.any():
        event("output carries DACA rows")
    if cuban_haitian.any():
        event("output carries Cuban/Haitian entrants")
    expected = np.select(
        [
            facts.ssn == "CITIZEN",
            facts.ssn == "NONE",
            facts.humanitarian,
            daca,
            cuban_haitian,
        ],
        ["CITIZEN", "UNDOCUMENTED", facts.status, "DACA", "CUBAN_HAITIAN_ENTRANT"],
        default="LEGAL_PERMANENT_RESIDENT",
    )
    np.testing.assert_array_equal(facts.status, expected)
    assert not (facts.humanitarian & (daca | cuban_haitian)).any()


@_STAGE
@given(_stage_cases())
def test_indicator_pool_is_exactly_the_noncitizens_with_a_documented_indicator(
    case,
) -> None:
    """Module docstring, step 2: "Non-citizens with any indicator of authorized
    status — pre-1982 IRCA-cohort arrival, ... — move to
    ``OTHER_NON_CITIZEN``", and nothing later moves them (the draws and the
    spill only rewrite code 0). No other row lands there. (The docstring also
    lists naturalization-eligibility, which the code cannot fire for a
    non-citizen; see ``_NEUTRAL_INDICATORS``.)"""

    output = _run(case)
    facts = _facts(case.person, output, case.time_period)
    np.testing.assert_array_equal(
        facts.ssn == "OTHER_NON_CITIZEN", facts.noncitizen & facts.indicator
    )


@_STAGE
@given(_stage_cases())
def test_humanitarian_labels_respect_exclusions_pools_and_draw_masks(case) -> None:
    """Module docstring, step 5: "Cuba/Haiti-born persons are excluded from
    every humanitarian draw ..., as is the DACA statutory cohort"; draws are
    "mutually exclusive"; "Draws from the residual pool (TPS and parole only)
    flip ``ssn_card_type`` ``NONE`` to ``NON_CITIZEN_VALID_EAD``";
    ``_humanitarian_draw_candidates``: "``REFUGEE``/``ASYLEE``/
    ``DEPORTATION_WITHHELD`` draw from the indicator-documented pool only".
    The public ``us_immigration_humanitarian_draw_mask`` partitions the
    humanitarian rows by draw, and an explicit zero target emits nothing
    (``HumanitarianDraw``: "Zero ... means the category is explicitly not
    imputed")."""

    output = _run(case)
    facts = _facts(case.person, output, case.time_period)
    if facts.humanitarian.any():
        event("output carries humanitarian rows")
    assert not (facts.humanitarian & ~facts.noncitizen).any()
    assert not (facts.humanitarian & facts.cuban_haitian_born).any()
    assert not (facts.humanitarian & facts.daca_cohort).any()
    indicator_pool = np.isin(facts.status, list(_INDICATOR_POOL_STATUSES))
    assert (facts.ssn[indicator_pool] == "OTHER_NON_CITIZEN").all()
    temporary = facts.humanitarian & ~indicator_pool
    if (temporary & (facts.ssn == "NON_CITIZEN_VALID_EAD")).any():
        event("a residual row drew parole/TPS")
    assert np.isin(
        facts.ssn[temporary], ("OTHER_NON_CITIZEN", "NON_CITIZEN_VALID_EAD")
    ).all()

    frame = _frame(output, np.ones(len(output)))
    covered = np.zeros(len(output), dtype=int)
    for draw in case.controls.humanitarian:
        mask = immigration.us_immigration_humanitarian_draw_mask(
            frame, draw, time_period=case.time_period
        )
        if draw.target <= 0:
            assert not mask.any(), draw.label
        covered += mask
    assert (covered <= 1).all()
    np.testing.assert_array_equal(covered == 1, facts.humanitarian)


@_STAGE
@given(_stage_cases())
def test_humanitarian_rows_sit_inside_the_asec_windows_the_comments_state(
    case,
) -> None:
    """Independent of ``_humanitarian_draw_candidates``: every emitted parole,
    TPS and asylee row satisfies the ASEC origin/arrival windows transcribed
    above from the module's comments (``_PAROLE_DOCUMENTED_WINDOWS``,
    ``_TPS_DOCUMENTED_MAX_ARRIVAL_CODE``, ``_TPS_DOCUMENTED_EXCLUDED_BIRTHS``,
    ``_ASYLEE_DOCUMENTED_MIN_ARRIVAL_CODE``). Refugee and withholding windows
    have no numeric statement in a comment and are not checked here."""

    output = _run(case)
    status = output["immigration_status_str"].to_numpy()
    birth = case.person["PENATVTY"].to_numpy()
    arrival = case.person["PEINUSYR"].to_numpy()

    parole = status == "PAROLED_ONE_YEAR"
    assert set(birth[parole].tolist()) <= set(_PAROLE_DOCUMENTED_WINDOWS)
    for origin, window in _PAROLE_DOCUMENTED_WINDOWS.items():
        rows = parole & (birth == origin)
        if rows.any():
            event("parole rows emitted")
        if ((birth == origin) & ~np.isin(arrival, list(window))).any():
            event("a parole-origin row sits outside its window")
        assert np.isin(arrival[rows], list(window)).all(), origin

    tps = status == "TPS"
    assert not np.isin(birth[tps], list(_TPS_DOCUMENTED_EXCLUDED_BIRTHS)).any()
    assert (arrival[tps] >= 1).all()
    for origin, latest in _TPS_DOCUMENTED_MAX_ARRIVAL_CODE.items():
        rows = tps & (birth == origin)
        if rows.any():
            event("legacy-designation TPS rows emitted")
        if ((birth == origin) & (arrival > latest)).any():
            event("a legacy TPS-origin row arrived after its cutoff")
        assert (arrival[rows] <= latest).all(), origin

    asylee = status == "ASYLEE"
    if asylee.any():
        event("asylee rows emitted")
    assert (arrival[asylee] >= _ASYLEE_DOCUMENTED_MIN_ARRIVAL_CODE).all()


@_STAGE
@given(_stage_cases())
def test_each_humanitarian_draw_reaches_its_target_or_exhausts_its_pool(
    case,
) -> None:
    """Step 5 draws each status "to a manifest-cited weighted stock target",
    "in that order", and "a person two cohorts could claim takes the earlier
    status" (``_assign_humanitarian_statuses``: "a person marked by an earlier
    draw is out of every later candidate pool"). The band constant adds: "The
    draw forces the target when candidates suffice". Replaying the candidate
    pools in draw order, each draw's rows reach its target and overshoot it by
    less than the last-drawn row, or take the whole pool when it is short; a
    zero target takes nothing; and the draws account for every humanitarian
    row. The pools come from the module's ``_humanitarian_draw_candidates``,
    so this checks selection against target, not the windows themselves."""

    output = _run(case)
    facts = _facts(case.person, output, case.time_period)
    tolerance = _tolerance(facts.weights)
    profile = immigration._source_aware_immigration_profile(
        case.person, time_period=case.time_period
    )
    pre_draw = _pre_draw_codes(facts.ssn)
    marked = np.zeros(len(output), dtype=bool)
    for draw in case.controls.humanitarian:
        pool = (
            immigration._humanitarian_draw_candidates(
                draw, ssn_codes=pre_draw, profile=profile
            )
            & ~marked
        )
        chosen = pool & (facts.status == draw.status)
        weight = float(facts.weights[chosen].sum())
        available = float(facts.weights[pool].sum())
        if draw.target <= 0:
            assert not chosen.any(), draw.label
        elif available >= draw.target + tolerance:
            event("a draw's pool sufficed")
            assert weight >= draw.target - tolerance, draw.label
            draws = immigration._stable_person_draws(
                case.person, seed=case.seed, salt=draw.salt
            )
            last = chosen & (draws == draws[chosen].max())
            assert (
                weight - float(facts.weights[last].sum()) < draw.target + tolerance
            ), draw.label
        elif available < draw.target - tolerance:
            event("a draw's pool was short")
            np.testing.assert_array_equal(chosen, pool, err_msg=draw.label)
        marked |= chosen
    np.testing.assert_array_equal(marked, facts.humanitarian)


@_STAGE
@given(_stage_cases(), st.data())
def test_humanitarian_draws_are_sequential_and_precede_the_spill(case, data) -> None:
    """``HUMANITARIAN_STATUS_CATEGORIES``: "Draws are sequential and mutually
    exclusive"; ``_assign_humanitarian_statuses`` "Runs after the legal-status
    indicators and before the EAD worker/student spill". So retargeting every
    category from some point on, and changing the worker and student controls,
    leaves the rows of every earlier category unchanged."""

    categories = immigration.HUMANITARIAN_STATUS_CATEGORIES
    cut = data.draw(st.integers(0, len(categories)), label="first retargeted")
    scale = max(1.0, float(case.person["person_weight"].sum()))
    pools = _pool_masses(case.person, case.time_period)
    residual = _residual_masses(case.person)
    later = set(categories[cut:])
    targets = {
        label: (
            data.draw(_target(pools[label], scale), label=label)
            if label.split(":", 1)[0] in later
            else target
        )
        for label, target in case.targets.items()
    }
    changed = replace(
        case,
        workers=data.draw(_control(residual["workers"], scale), label="workers"),
        students=data.draw(_control(residual["students"], scale), label="students"),
        targets=targets,
    )
    first = _run(case)["immigration_status_str"].to_numpy()
    second = _run(changed)["immigration_status_str"].to_numpy()
    for category in categories[:cut]:
        status = immigration._HUMANITARIAN_STATUS_BY_CATEGORY[category]
        if (first == status).any():
            event("an earlier category emitted rows")
        np.testing.assert_array_equal(
            first == status, second == status, err_msg=category
        )


@_STAGE
@given(_stage_cases(integer_weights=True), st.data())
def test_stage_is_bit_reproducible_and_row_order_equivariant(case, data) -> None:
    """Module docstring: "reruns are bit-reproducible without global RNG
    state". Row-order equivariance is deduced: draws are "keyed by the
    person's stable source identity" and every selection is a draw threshold
    (integer weights keep the weighted sums order-independent)."""

    output = _run(case)
    pd.testing.assert_frame_equal(_run(case), output)
    order = list(data.draw(st.permutations(range(len(case.person)))))
    shuffled = replace(case, person=case.person.iloc[order].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        _run(shuffled),
        output.iloc[order].reset_index(drop=True),
    )


@_STAGE
@given(_stage_cases(max_size=16), st.data())
def test_support_clones_share_their_source_persons_labels(case, data) -> None:
    """Module docstring: draws keyed by
    ``source_year``/``source_household_id``/``source_person_id`` mean
    "support-channel clones of one source person always receive the same
    status"; ``_select_weight_to_target``: "a clone pair is never split".
    Clones copy their source person's survey record but may carry any
    weight."""

    source = case.person.assign(
        source_year=2024,
        source_household_id=case.person["person_id"] + 100,
        source_person_id=case.person["person_id"],
    )
    copies = data.draw(
        st.lists(st.integers(0, len(source) - 1), min_size=1, max_size=16),
        label="cloned rows",
    )
    clones = source.iloc[copies].copy()
    clones["person_weight"] = data.draw(
        st.lists(_WEIGHT, min_size=len(copies), max_size=len(copies)),
        label="clone weights",
    )
    person = pd.concat([source, clones], ignore_index=True)
    person["person_id"] = np.arange(1, len(person) + 1)
    output = _derive(
        person, case.parameters, seed=case.seed, time_period=case.time_period
    )
    grouped = output.groupby("source_person_id")[
        list(immigration.US_IMMIGRATION_OUTPUT_COLUMNS)
    ].nunique()
    assert (grouped == 1).all().all()


@_FRAME
@given(_stage_cases(max_size=16), st.floats(0.25, 4.0), st.data())
def test_frame_entry_point_matches_the_manifest_handler_and_is_idempotent(
    case, weight_scale, data
) -> None:
    """``with_us_immigration_inputs`` runs "the ``immigration_status``
    manifest stage over a US frame" and returns "A new frame whose person
    table carries ``ssn_card_type`` and ``immigration_status_str``";
    ``person_weight_scale`` is a "Stage-only multiplier for person design
    weights"; and "Existing output columns are preserved, making the
    transform idempotent". Differential: with non-monotone person ids, the
    frame entry point labels every person exactly as the packaged manifest
    handler does on design weight x scale, leaves the frame's weights and
    every other column and table unchanged, and a second call (any seed or
    period) returns the frame as it is."""

    ids = data.draw(st.permutations(range(1, len(case.person) + 1)), label="ids")
    person = case.person.assign(person_id=np.asarray(ids, dtype=np.int64))
    # A Frame refuses all-zero weights, which the handler alone accepts.
    weights = (person["person_weight"].to_numpy(dtype=np.float64) + 1.0) * 1e3
    frame = _frame(person, weights)
    result = immigration.with_us_immigration_inputs(
        frame,
        seed=case.seed,
        time_period=case.time_period,
        person_weight_scale=weight_scale,
    )
    packaged = [
        operation
        for operation in immigration.us_immigration_stage_spec().operations
        if operation.kind == "derive_immigration_status"
    ]
    assert len(packaged) == 1
    expected = _derive(
        frame.table("person").assign(person_weight=weights * weight_scale),
        packaged[0].parameters,
        seed=case.seed,
        time_period=case.time_period,
    )
    labelled = result.table("person")
    columns = list(immigration.US_IMMIGRATION_OUTPUT_COLUMNS)
    for column in columns:
        np.testing.assert_array_equal(
            labelled[column].to_numpy(), expected[column].to_numpy(), err_msg=column
        )
    pd.testing.assert_frame_equal(labelled.drop(columns=columns), frame.table("person"))
    for entity in US_SCHEMA.group_entities:
        pd.testing.assert_frame_equal(result.table(entity), frame.table(entity))
    np.testing.assert_array_equal(
        np.asarray(result.resolve_weights("household").values),
        np.asarray(frame.resolve_weights("household").values),
    )
    again = immigration.with_us_immigration_inputs(
        result,
        seed=case.seed + 1,
        time_period=case.time_period + 1,
        person_weight_scale=2.0,
    )
    pd.testing.assert_frame_equal(again.table("person"), labelled)


#: Composition-gate failures that judge plausibility of the weighted totals
#: (non-citizen share, Pew anchor band, humanitarian bands, a small table
#: that happens to be all one value). Every other failure is structural.
_PLAUSIBILITY_FAILURES = (
    "constant value",
    "non-citizen weighted share",
    "emergent Pew-defined unauthorized population",
    "x the cited stock target",
)


def _structural_failures(gate) -> list[str]:
    return [
        failure
        for failure in gate.failures
        if not any(fragment in failure for fragment in _PLAUSIBILITY_FAILURES)
    ]


def _binary_weights(n: int) -> np.ndarray:
    """Distinct powers of two: a weighted total then names its rows exactly."""

    return 2.0 ** np.arange(n, dtype=np.float64)


def _rows_from_binary_total(total: float, n: int) -> np.ndarray:
    bits = int(round(total))
    assert float(bits) == total
    return np.asarray([(bits >> index) & 1 for index in range(n)], dtype=bool)


def _person_row(**overrides) -> dict:
    """One ``_asec_row``-shaped record: a US-born citizen adult by default."""

    row = {
        "PRCITSHP": 1,
        "PENATVTY": 57,
        "PEINUSYR": 0,
        "A_AGE": 30,
        "A_MARITL": 7,
        "A_SPOUSE": 0,
        "A_HSCOL": 0,
        "A_LFSR": 7,
        "indicator": None,
        "person_weight": 1.0,
    }
    return row | overrides


#: A worker who arrived in 2007 (PEINUSYR 20) and is 33 entered at 16 when
#: measured in 2024 but at 15 in 2025, so the stage at 2025 emits DACA once
#: the worker spill gives the row an EAD. A gate that re-derived the cohort
#: at 2024 (the defect fixed in 5a89fed1e) flags that row.
_PERIOD_SENSITIVE_DACA_CASE = _StageCase(
    person=_asec_table(
        [
            _person_row(),
            _person_row(PRCITSHP=5, PENATVTY=303, PEINUSYR=20, A_AGE=33, A_LFSR=1),
        ]
    ),
    workers=0.5,
    students=1.0,
    anchor=1.0,
    targets=dict.fromkeys(_DRAW_LABELS, 0.0),
    seed=0,
    time_period=2025,
)


@_STAGE
@given(_stage_cases())
@example(_PERIOD_SENSITIVE_DACA_CASE)
def test_stage_output_satisfies_the_composition_gates_structural_contract(
    case,
) -> None:
    """Differential against ``us_immigration_composition_gate``, which fails
    "when a value falls outside the engine enum domain, when the two columns
    disagree about citizenship or undocumented status", on source-evidence and
    Cuban/Haitian/DACA cohort violations, on humanitarian rows outside every
    draw's cohort, and when "an explicit zero target" emits. Its
    ``time_period`` "must be the period the stage ran for", so the gate runs
    at the stage's period (several periods are drawn). The stage and the gate
    encode those contracts separately; stage output must pass all of them
    (the weighted plausibility bands are not structural)."""

    output = _run(case)
    event(f"time period {case.time_period}")
    gate = immigration.us_immigration_composition_gate(
        _frame(output, _binary_weights(len(output))),
        time_period=case.time_period,
        controls=case.controls,
    )
    assert _structural_failures(gate) == []
    assert gate.details["evidence_derived_status_compatibility"]["invalid_rows"] == 0


@_STAGE
@given(_stage_cases())
def test_spill_projection_selects_exactly_the_gates_pew_universe(case) -> None:
    """Differential: ``_pew_unauthorized_projection_mask`` "Project[s] final
    membership in Pew's broad unauthorized universe" that the composition
    gate then measures (``_PEW_UNAUTHORIZED_STATUS_VALUES`` plus residual
    Cuban/Haitian EAD rows). On final stage output the two definitions select
    the same rows; power-of-two gate weights make the gate's total name its
    rows."""

    output = _run(case)
    facts = _facts(case.person, output, case.time_period)
    gate = immigration.us_immigration_composition_gate(
        _frame(output, _binary_weights(len(output))),
        time_period=case.time_period,
        controls=case.controls,
    )
    gate_rows = _rows_from_binary_total(
        gate.details["pew_unauthorized_population"], len(output)
    )
    projection = immigration._pew_unauthorized_projection_mask(
        ssn_codes=_codes(facts.ssn),
        humanitarian_marks=np.where(facts.humanitarian, facts.status, ""),
        retained_ead_cohort=facts.retained,
    )
    if (projection & (facts.ssn != "NONE")).any():
        event("Pew universe includes non-NONE rows")
    np.testing.assert_array_equal(projection, gate_rows)


@_STAGE
@given(_stage_cases())
def test_pew_margins_close_whenever_the_residual_pool_can_close_them(case) -> None:
    """Module docstring, step 3: workers and students "spill to
    ``NON_CITIZEN_VALID_EAD`` in deterministic seeded order until the broad
    reported universes match their controls"; the worker spill "runs last and
    is therefore authoritative"; and "a count already at or below its control
    spills nothing". Each margin ends at or below its control unless every
    removable (residual, non-retained) row in it was spilled, and no row is
    spilled when both margins start below their controls."""

    output = _run(case)
    facts = _facts(case.person, output, case.time_period)
    tolerance = _tolerance(facts.weights)
    pew = immigration._pew_unauthorized_projection_mask(
        ssn_codes=_codes(facts.ssn),
        humanitarian_marks=np.where(facts.humanitarian, facts.status, ""),
        retained_ead_cohort=facts.retained,
    )
    residual = (
        facts.noncitizen
        & ~facts.humanitarian
        & np.isin(facts.ssn, ("NONE", "NON_CITIZEN_VALID_EAD"))
    )
    spilled = residual & (facts.ssn == "NON_CITIZEN_VALID_EAD")
    initial: dict[str, float] = {}
    for name, scope, target in (
        ("students", facts.student, case.students),
        ("workers", facts.worker, case.workers),
    ):
        final = float(facts.weights[scope & pew].sum())
        removable = scope & residual & ~facts.retained
        if (scope & spilled).any():
            event(f"the {name} margin spilled rows")
        if final <= target + tolerance and (removable & ~spilled).any():
            event(f"the {name} margin closed with residual rows left")
        assert final <= target + tolerance or not (removable & ~spilled).any(), name
        initial[name] = final + float(
            facts.weights[scope & spilled & ~facts.retained].sum()
        )
    if (
        initial["students"] < case.students - tolerance
        and initial["workers"] < case.workers - tolerance
    ):
        assert not spilled.any()


# ---------------------------------------------------------------------------
# ACS transfer reconciliation
# ---------------------------------------------------------------------------

_ACS_ROW = st.fixed_dictionaries(
    {
        "CIT": st.sampled_from((1, 4, 5, 5, 5, 5)),
        # Biased toward the humanitarian origins so recipients can fill draws.
        "POBP": st.one_of(
            st.sampled_from(_BIRTH_CODES), st.sampled_from(_HUMANITARIAN_ORIGIN_CODES)
        ),
        "YOEP": st.one_of(st.integers(1975, 2024), st.integers(2016, 2024)),
        "age": st.integers(0, 85),
        "ssn_card_type": st.sampled_from(immigration.SSN_CARD_TYPE_VALUES),
        # The joint-QRF baseline never emits an evidence-constrained label.
        "immigration_status_str": st.sampled_from(
            ("CITIZEN", "LEGAL_PERMANENT_RESIDENT", "UNDOCUMENTED")
        ),
        "weight": st.one_of(
            st.integers(1, 5_000).map(float),
            st.floats(0.5, 5e5, allow_nan=False, allow_subnormal=False),
        ),
    }
)


@dataclass(frozen=True)
class _TransferCase:
    person: pd.DataFrame
    weights: np.ndarray
    mutable: np.ndarray
    controls: immigration.ImmigrationControls
    seed: int


@st.composite
def _transfer_cases(draw) -> _TransferCase:
    """ASEC rows labeled by the stage (immutable) stacked with ACS recipients
    carrying unconstrained donor labels (mutable), as in the ACS transfer.

    Reconciliation targets are feasible by construction, so the documented
    refusal ("candidate exhaustion is a hard error") never applies and any
    error is a finding. Each draw's target is at least half its immutable ASEC
    mass (so a draw with immutable rows never has an explicit-zero target)
    plus a share of the recipient mass that this draw, and no earlier draw,
    can claim; ``selected_once`` can only remove rows an earlier draw claims.
    Some recipients are ACS twins of ASEC rows (same origin, age and the
    ASEC bin's midpoint year), so a draw often has both an immutable
    contribution and recipients to select.
    """

    stage = draw(_stage_cases(min_size=4, max_size=16, time_periods=(_TIME_PERIOD,)))
    asec = _run(stage).assign(CIT=np.nan, POBP=np.nan, YOEP=np.nan, age=np.nan)
    acs_rows = draw(st.lists(_ACS_ROW, min_size=2, max_size=12))
    for index in draw(
        st.lists(st.integers(0, len(asec) - 1), max_size=8), label="ACS twins"
    ):
        source = asec.iloc[index]
        noncitizen = int(source["PRCITSHP"]) == 5
        acs_rows.append(
            {
                "CIT": 5 if noncitizen else 1,
                "POBP": int(source["PENATVTY"]),
                "YOEP": immigration._ARRIVAL_YEAR_MIDPOINTS.get(
                    int(source["PEINUSYR"]), _TIME_PERIOD
                ),
                "age": int(source["A_AGE"]),
                "ssn_card_type": draw(
                    st.sampled_from(
                        ("OTHER_NON_CITIZEN", "NONE", "NON_CITIZEN_VALID_EAD")
                        if noncitizen
                        else ("CITIZEN",)
                    )
                ),
                "immigration_status_str": draw(
                    st.sampled_from(("LEGAL_PERMANENT_RESIDENT", "UNDOCUMENTED"))
                    if noncitizen
                    else st.just("CITIZEN")
                ),
                "weight": draw(st.integers(1, 5_000).map(float)),
            }
        )
    acs = pd.DataFrame(acs_rows)
    acs = acs.rename(columns={"weight": "person_weight"}).assign(
        PRCITSHP=np.nan, PENATVTY=np.nan, PEINUSYR=np.nan, A_AGE=np.nan
    )
    person = pd.concat([asec, acs], ignore_index=True)
    person["person_id"] = np.arange(1, len(person) + 1)
    weights = person.pop("person_weight").to_numpy(dtype=np.float64)
    mutable = np.arange(len(person)) >= len(asec)

    frame = _frame(person, weights)
    profile = immigration._source_aware_immigration_profile(
        person, time_period=_TIME_PERIOD
    )
    # Pair repair turns a recipient non-citizen's CITIZEN code into
    # OTHER_NON_CITIZEN and leaves every other non-citizen code alone; no
    # citizen is ever a humanitarian candidate.
    codes = immigration._ssn_name_codes(person)
    codes = np.where(~profile.is_citizen & (codes == 1), 3, codes)
    claimed = np.zeros(len(person), dtype=bool)
    targets: dict[str, float] = {}
    for control in stage.controls.humanitarian:
        immutable_mass = float(
            weights[
                immigration.us_immigration_humanitarian_draw_mask(
                    frame, control, time_period=_TIME_PERIOD
                )
                & ~mutable
            ].sum()
        )
        candidate_codes = (
            np.where(codes == 2, 0, codes)
            if control.category in {"paroled_one_year", "tps"}
            else codes
        )
        pool = mutable & immigration._humanitarian_draw_candidates(
            control, ssn_codes=candidate_codes, profile=profile
        )
        exclusive = float(weights[pool & ~claimed].sum())
        claimed |= pool
        immutable_share = draw(
            st.one_of(st.just(1.0), st.floats(0.5, 1.0)), label="immutable share"
        )
        exclusive_share = draw(
            st.one_of(st.just(0.0), st.just(1.0), st.floats(0.0, 1.0)),
            label="exclusive share",
        )
        targets[control.label] = (
            immutable_share * immutable_mass + exclusive_share * exclusive
        )
    return _TransferCase(
        person=person,
        weights=weights,
        mutable=mutable,
        controls=immigration._controls_from_parameters(
            _manifest_parameters(
                workers=stage.workers,
                students=stage.students,
                anchor=stage.anchor,
                targets=targets,
            )
        ),
        seed=draw(st.integers(0, 2**31 - 1)),
    )


def _reconcile(case: _TransferCase):
    return immigration.reconcile_us_immigration_humanitarian_transfer(
        case.person,
        weights=case.weights,
        mutable_rows=case.mutable,
        seed=case.seed,
        time_period=_TIME_PERIOD,
        controls=case.controls,
    )


@_TRANSFER
@given(_transfer_cases())
def test_reconciliation_output_satisfies_the_gates_structural_contract(case) -> None:
    """Differential: ``reconcile_us_immigration_humanitarian_transfer``'s own
    final checks restate the gate's citizenship/``NONE``⇔``UNDOCUMENTED``
    pairing and Cuban/Haitian/DACA cohort contracts; the independently coded
    ``us_immigration_composition_gate`` must find no structural failure in
    the reconciled frame. Reconciliation is deterministic. Immutable rows are
    compared too, but the function already raises "changed immutable ASEC
    rows" before returning otherwise, so that comparison confirms the guard
    rather than adding evidence."""

    result, receipt = _reconcile(case)
    if any(
        entry["selected_recipient_population"] > 0
        for entry in receipt["draws"].values()
    ):
        event("reconciliation selected recipients")
    immutable = ~case.mutable
    for column in immigration.US_IMMIGRATION_OUTPUT_COLUMNS:
        np.testing.assert_array_equal(
            result.loc[immutable, column].to_numpy(),
            case.person.loc[immutable, column].to_numpy(),
        )
    gate = immigration.us_immigration_composition_gate(
        _frame(result, case.weights),
        time_period=_TIME_PERIOD,
        controls=case.controls,
    )
    assert _structural_failures(gate) == []
    again_result, again_receipt = _reconcile(case)
    pd.testing.assert_frame_equal(again_result, result)
    assert again_receipt == receipt


@_TRANSFER
@given(_transfer_cases())
def test_reconciliation_receipts_obey_their_accounting_identities(case) -> None:
    """Each draw "receives the residual of its national target after its
    compatible, immutable contribution, in manifest order", and the receipt
    records it.

    Independent evidence: ``achieved_population`` is recomputed from the
    post-selection emitted mask, separately from the immutable and selected
    masks, and must equal immutable + selected; a zero target selects
    nothing; the selection never exceeds the eligible pool.

    Receipt consistency only (not independent evidence): the function raises
    ``RuntimeError`` before returning a receipt whose selection misses its
    residual by more than its threshold tie mass or whose achieved error
    exceeds the discrete bound, and it hard-codes
    ``within_residual_discrete_weight_bound``. Those assertions confirm the
    receipt records the quantities it checked."""

    _, receipt = _reconcile(case)
    tolerance = float(receipt["floating_tolerance"])
    assert receipt["selection_order"] == [
        draw.label for draw in case.controls.humanitarian
    ]
    assert receipt["mutable_rows"] == int(case.mutable.sum())
    for draw in case.controls.humanitarian:
        entry = receipt["draws"][draw.label]
        target = float(draw.target)
        immutable = entry["immutable_population"]
        selected = entry["selected_recipient_population"]
        achieved = entry["achieved_population"]
        tie = entry["selection_threshold_tie_population"]
        if selected > 0 and immutable > 0:
            event("a draw combined immutable and selected rows")
        if immutable > target:
            event("immutable rows overshoot a target")
        # Independent identities.
        assert achieved == pytest.approx(immutable + selected, abs=tolerance)
        assert selected <= entry["eligible_recipient_population"] + tolerance
        if target <= 0:
            assert selected == 0.0
        # Receipt consistency with the function's own guards.
        assert entry["target"] == target
        assert entry["residual_target"] == pytest.approx(
            max(0.0, target - immutable), abs=tolerance
        )
        assert abs(selected - entry["residual_target"]) <= tie + tolerance
        assert abs(achieved - target) <= (
            max(0.0, immutable - target) + tie + tolerance
        )
        assert entry["within_residual_discrete_weight_bound"] is True
        assert entry["within_discrete_weight_bound"] is True


@_TRANSFER
@given(_transfer_cases())
def test_selection_replay_reproduces_the_reconciled_mutable_rows(case) -> None:
    """Differential: ``us_immigration_humanitarian_transfer_selection_masks``
    promises to "Replay the exact deterministic mutable-row selection for
    every draw" from the post-transfer frame. For every draw, its replayed
    mask equals the reconciled mutable rows that
    ``us_immigration_humanitarian_draw_mask`` reports as emitted (the
    comparison ``stacked_spine`` makes at release)."""

    result, _ = _reconcile(case)
    frame = _frame(result, case.weights)
    replay = immigration.us_immigration_humanitarian_transfer_selection_masks(
        frame,
        mutable_rows=case.mutable,
        seed=case.seed,
        time_period=_TIME_PERIOD,
        controls=case.controls,
    )
    assert list(replay) == [draw.label for draw in case.controls.humanitarian]
    for draw in case.controls.humanitarian:
        emitted = immigration.us_immigration_humanitarian_draw_mask(
            frame, draw, time_period=_TIME_PERIOD
        )
        if (emitted & case.mutable).any():
            event("a replayed draw selected recipients")
        np.testing.assert_array_equal(
            replay[draw.label], emitted & case.mutable, err_msg=draw.label
        )
