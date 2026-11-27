"""Which ``ledger_filter_*`` metadata the US materializers act on.

The national release tool (``tools/build_us_fiscal_refresh_release.py``)
materializes each compiled target spec. These rules say how its ``irs_soi``
slice reads a spec's AGI band, filing status and EITC child count, and which
restated ``ledger_filter_*`` keys a spec may carry. microcosm#1728 moved them
here from the tool, unchanged, so that the ACS local district surface
(``tools/us_acs_local_cd_surface.py``) reads them without importing the tool.
The tool imports them back under the same names. Function references in the
comments below (``_materialize_target_frame`` and others) are the tool's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from microcosm.build.us_runtime.fiscal_targets import (
    AGE_BOUND_STAMP_FROM_CONSTRAINT_ROWS,
    AGE_BOUND_STAMP_SOURCE_KEY,
)
from microcosm.build.us_runtime.fsa_student_loan_targets import (
    FSA_BALANCE_LOWER_BOUND_KEY,
    FSA_BALANCE_UPPER_BOUND_KEY,
    FSA_MATERIALIZER,
)

#: Restated constraints, a third class distinct from both the supported and
#: the identity ledger filter keys (``SUPPORTED_LEDGER_FILTER_METADATA_KEYS``
#: and ``IDENTITY_LEDGER_FILTER_METADATA_KEYS`` in the national release tool): a
#: labelled feed re-expresses a fact's universe constraint in the Ledger's own
#: concept vocabulary (``us:statutes/26/62#adjusted_gross_income_lower_bound``)
#: alongside the compiled metadata the materializer actually slices on
#: (``agi_lower_bound``). These keys are NOT inert — they restrict the
#: microdata — so they can never join the supported set outright: a future
#: feed whose labelled bound disagreed with the compiled one would then be
#: silently ignored, materializing a wider population than the published cell
#: covers, which is the failure the guard exists to stop. Each key here is
#: accepted only per spec, and only when that spec also carries the filter the
#: materializer applies and the two select the same population; a restatement
#: that disagrees, or that has no compiled counterpart, stays fatal and the
#: refusal names the values. Values are the restated-constraint rule that
#: decides the comparison (see ``_restated_ledger_filter_refusal``).
RESTATED_LEDGER_FILTER_CONCEPTS = {
    "us:statutes/26/62#adjusted_gross_income": "agi_band",
    "us.tax.earned_income_credit_qualifying_children": "eitc_child_count",
    "age": "age_band",
    "federal_student_loan_balance": "student_loan_balance_band",
}


#: Bound-side suffixes a restated constraint key may carry. A key with no
#: suffix restates an exact value of the concept.
RESTATED_LEDGER_FILTER_BOUND_SIDES = (
    ("_lower_bound", "lower"),
    ("_upper_bound", "upper"),
)


#: Compiled metadata the materializer slices the AGI band on, by bound
#: side: the ``irs_soi`` loop in :func:`_materialize_target_frame` reads
#: both through :func:`_as_bound` into one half-open mask. A restated AGI
#: bound is accepted only against its own side; the sibling side is judged
#: by its own key.
RESTATED_AGI_BAND_COMPILED_KEYS = {
    "lower": "agi_lower_bound",
    "upper": "agi_upper_bound",
}


#: Qualifying-child counts probed when deciding whether a restated EITC
#: child-count bound selects the same returns as the compiled filter.
#: :func:`_eitc_child_count_mask` resolves every compiled value to ``== 0``,
#: ``== 1``, ``== 2`` or ``>= 3``, and a compared restatement is ``>= n`` or
#: ``== n`` (an upper bound is refused before any comparison; see
#: :data:`RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES`); evaluating both over
#: 0..16 separates every such pair, so equal masks over the probe mean equal
#: populations. Counts above the probe are refused rather than compared.
RESTATED_EITC_CHILD_COUNT_PROBE_MAX = 16


#: Bound sides on which a restated EITC qualifying-child key is refused
#: outright, whatever its value and whatever the spec compiles. What an upper
#: bound selects turns on the Ledger's operator — ``< 1`` is the childless
#: returns, ``<= 1`` adds the one-child returns — and that operator is
#: unconfirmed for count constraints. The metadata key does not settle it:
#: :func:`microcosm.build.ledger_targets._constraint_bound_filters` stamps a
#: ``<`` row as ``_upper_bound`` (``<=`` becomes ``_upper_bound_inclusive``,
#: which no restated concept names, so it is refused by bare key), but the
#: dimension stamp in ``_ledger_metadata`` runs first and wins through
#: ``setdefault``, and can carry the same key with no operator at all. So no
#: reading is guessed until the Ledger confirms one (Max, 2026-09-22). Lower
#: and exact restatements stay under the agreement rule.
RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES = frozenset({"upper"})


#: Compiled metadata the materializer slices a person-age band on, by bound
#: side. Both age paths in :func:`_materialize_target_frame` read these
#: through :func:`_as_bound` into one half-open mask ``lower <= age < upper``
#: — the ``population_age`` loop in :func:`_population_age_household_values`
#: and the age-banded ``policyengine_variable`` branch (the SSA SSI
#: recipients-by-age counts) — so a restated age bound is judged, like an AGI
#: bound, against its own side only.
#:
#: Numeric equality of the edges is equality of populations only for a
#: restated key whose operator is the mask's: ``age < 10`` and ``age <= 10``
#: differ by everyone aged exactly ten, so the operator decides.
#: :func:`microcosm.build.ledger_targets._constraint_bound_filters` writes
#: ``age_lower_bound`` only for a ``>=`` row and ``age_upper_bound`` only for
#: a ``<`` row, the two operators of the materializer's mask. A ``>`` or
#: ``<=`` row is stamped ``age_lower_bound_exclusive`` /
#: ``age_upper_bound_inclusive``, which no restated concept names, so it stays
#: refused by bare key — as it must, because the compiled bound comes from
#: ``us_runtime.fiscal_targets._age_bounds``, which drops the operator: a
#: ``<= 4`` row compiles to ``age_upper_bound=4`` and the materializer's
#: ``age < 4`` would leave out the four-year-olds the published cell counts.
#: But the same ``ledger_filter_age_{lower,upper}_bound`` key can also be a
#: dimension's, which carries no operator (the ambiguity that keeps restated
#: qualifying-child upper bounds refused), so the key name alone does not
#: settle it; the compile's attestation does
#: (:data:`microcosm.build.us_runtime.fiscal_targets.AGE_BOUND_STAMP_SOURCE_KEY`).
RESTATED_AGE_BAND_COMPILED_KEYS = {
    "lower": "age_lower_bound",
    "upper": "age_upper_bound",
}


#: Materializers that apply the compiled age band, and so the only ones for
#: which ignoring an agreeing age restatement changes nothing. Both are read
#: in :func:`_materialize_target_frame`: ``population_age`` through
#: :func:`_population_age_household_values`, and ``policyengine_variable``
#: through its age-banded branch, entered whenever either compiled age key is
#: present. Every other materializer (the ``irs_soi`` slice, the direct
#: household-variable measures) never reads an age bound, so a restated age
#: bound there would be silently ignored however well it agreed with metadata
#: the materializer also ignores; it is refused instead.
RESTATED_AGE_BAND_MATERIALIZERS = frozenset(
    {"population_age", "policyengine_variable", FSA_MATERIALIZER}
)


#: Compiled edges of a Federal Student Aid debt-size band, by bound side, and
#: the one materializer that slices on them
#: (:func:`microcosm.build.us_runtime.fsa_student_loan_targets.fsa_student_loan_household_values`,
#: ``lower <= balance < upper`` on the person's own plus Parent PLUS balance,
#: microcosm#1074). A restated ``federal_student_loan_balance`` bound on any
#: other materializer would be ignored, so it refuses.
RESTATED_STUDENT_LOAN_BALANCE_BAND_COMPILED_KEYS = {
    "lower": FSA_BALANCE_LOWER_BOUND_KEY,
    "upper": FSA_BALANCE_UPPER_BOUND_KEY,
}


#: The exact-value age key the dimension stamp writes when ``age`` is itself
#: a dimension of the fact.
#: :func:`microcosm.build.ledger_targets._constraint_bound_filters` skips
#: every constraint row whose variable is already a dimension key, so on
#: such a fact an ``age_{lower,upper}_bound`` restatement cannot have come
#: from an operator-checked ``>=``/``<`` row — only from a dimension of that
#: name, which carries no operator at all. A restated age bound on a spec
#: carrying this key is refused, whatever its value, including ``all``.
#: ``_ledger_metadata`` stamps a dimension only when its value is not
#: ``None`` and then drops empty values, so an ``age`` dimension valued
#: ``None`` or ``""`` leaves no key here; the compile's attestation
#: (:data:`AGE_BOUND_STAMP_SOURCE_KEY`, which checks dimension keys whatever
#: their value) is what refuses that case. The pinned feeds have no age
#: dimension of any kind.
RESTATED_AGE_DIMENSION_KEY = "ledger_filter_age"


SUPPORTED_SOI_LEDGER_FILTERS = frozenset(
    {
        "ledger_filter_income_range",
        "ledger_filter_filing_status",
        "ledger_filter_eitc_child_count",
    }
)


def _soi_eitc_child_count_filter(metadata: Mapping[str, str]) -> str | None:
    explicit = metadata.get("ledger_filter_eitc_child_count")
    if explicit and explicit.strip().lower() != "all":
        return explicit
    record_set = metadata.get("ledger_layout_record_set_id", "")
    if record_set.endswith(".eitc_by_agi_children.no_qualifying_children"):
        return "0"
    if record_set.endswith(".eitc_by_agi_children.one_qualifying_child"):
        return "1"
    if record_set.endswith(".eitc_by_agi_children.two_qualifying_children"):
        return "2"
    if record_set.endswith(".eitc_by_agi_children.three_or_more_qualifying_children"):
        return "3plus"
    measure = metadata.get("source_measure_id", "")
    if measure.startswith("eitc_no_children_"):
        return "0"
    if measure.startswith("eitc_one_child_"):
        return "1"
    if measure.startswith("eitc_two_children_"):
        return "2"
    if measure.startswith("eitc_three_or_more_children_"):
        return "3plus"
    return None


def _soi_requires_positive_eitc_filter(metadata: Mapping[str, str]) -> bool:
    return (
        metadata.get("ledger_domain")
        == "individual_income_tax_returns_with_earned_income_credit"
    )


def _is_noop_ledger_filter_value(value: str) -> bool:
    return value.strip().lower().replace("_", " ") in {"", "all"}


def _unsupported_soi_ledger_filters(metadata: Mapping[str, str]) -> tuple[str, ...]:
    """Ledger filter keys the SOI slice does not act on, for one spec.

    A non-empty result drops the spec from SOI materialization silently in
    the ``irs_soi`` loop of :func:`_materialize_target_frame`, so an
    accepted restatement must clear here too — otherwise
    accepting it at the fatal guard would only move the spec from a refusal
    to a silent disappearance. A restatement that disagrees stays listed, and
    :func:`_assert_supported_ledger_filter_metadata` refuses it before the
    materializer ever reaches this skip.
    """

    return tuple(
        sorted(
            key
            for key, value in metadata.items()
            if key.startswith("ledger_filter_")
            and key not in SUPPORTED_SOI_LEDGER_FILTERS
            and not _is_noop_ledger_filter_value(str(value))
            and _restated_ledger_filter_refusal(str(key), str(value), metadata)
            is not None
        )
    )


def _eitc_child_count_mask(values: np.ndarray, filter_value: str) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    normalized = str(filter_value).strip().lower().replace("_", " ")
    if normalized in {"0", "none", "no children", "no qualifying children"}:
        return counts == 0
    if normalized in {"1", "one", "one child", "one qualifying child"}:
        return counts == 1
    if normalized in {"2", "two", "two children", "two qualifying children"}:
        return counts == 2
    if normalized in {
        "3",
        "3+",
        "3plus",
        "three",
        "three plus",
        "three or more",
        "three or more children",
        "three or more qualifying children",
    }:
        return counts >= 3
    raise ValueError(f"Unsupported EITC child-count filter {filter_value!r}.")


def _as_bound(value: str) -> float:
    if value == "-inf":
        return -math.inf
    if value == "inf":
        return math.inf
    return float(value)


def _restated_ledger_filter_concept(key: str) -> tuple[str, str | None]:
    """Split a ``ledger_filter_*`` key into its concept and bound side.

    ``ledger_filter_us:statutes/26/62#adjusted_gross_income_lower_bound``
    splits into ``("us:statutes/26/62#adjusted_gross_income", "lower")``; a
    key with no bound suffix keeps the whole concept and side ``None``,
    meaning it restates an exact value.
    """

    if not key.startswith("ledger_filter_"):
        return "", None
    concept = key[len("ledger_filter_") :]
    for suffix, side in RESTATED_LEDGER_FILTER_BOUND_SIDES:
        if concept.endswith(suffix) and len(concept) > len(suffix):
            return concept[: -len(suffix)], side
    return concept, None


def _restated_bound_value(value: str) -> float | None:
    try:
        return _as_bound(value.strip())
    except ValueError:
        return None


def _restated_count_value(value: str) -> float | None:
    try:
        count = float(value.strip())
    except ValueError:
        return None
    if not count.is_integer() or not 0 <= count <= RESTATED_EITC_CHILD_COUNT_PROBE_MAX:
        return None
    return count


def _restated_band_edge_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    compiled_key: str,
) -> str | None:
    """Judge one restated half-open band edge against the edge it restates.

    Shared by the AGI and age rules, whose materializers both slice
    ``lower <= x < upper`` on the compiled edge read through
    :func:`_as_bound`, so a restated edge is ignorable exactly when it parses
    to the same number. A missing compiled edge refuses: the age paths would
    read it as the open end and the ``irs_soi`` loop indexes it directly, and
    neither is the restated edge being checked.
    """

    compiled = metadata.get(compiled_key)
    if compiled is None:
        return f"{key}={value} restates a bound the spec does not compile: no {compiled_key}"
    restated = _restated_bound_value(value)
    applied = _restated_bound_value(str(compiled))
    if restated is None or applied is None or restated != applied:
        return f"{key}={value} disagrees with {compiled_key}={compiled}"
    return None


def _restated_agi_band_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side is None:
        return (
            f"{key}={value} restates an exact AGI, but the materializer "
            "slices a half-open AGI band and applies no exact-value AGI filter"
        )
    return _restated_band_edge_refusal(
        key, value, metadata, compiled_key=RESTATED_AGI_BAND_COMPILED_KEYS[side]
    )


def _restated_age_band_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side is None:
        return (
            f"{key}={value} restates an exact age, but the materializer "
            "slices a half-open age band and applies no exact-age filter"
        )
    if RESTATED_AGE_DIMENSION_KEY in metadata:
        return (
            f"{key}={value} restates an age bound on a fact whose dimensions "
            f"include age ({RESTATED_AGE_DIMENSION_KEY}="
            f"{metadata[RESTATED_AGE_DIMENSION_KEY]}), so the bound came from "
            "the dimension stamp, which carries no operator"
        )
    materializer = metadata.get("materializer")
    if materializer not in RESTATED_AGE_BAND_MATERIALIZERS:
        return (
            f"{key}={value} restates an age bound on a spec whose materializer "
            f"({materializer!r}) applies no age band"
        )
    stamp_source = metadata.get(AGE_BOUND_STAMP_SOURCE_KEY)
    if stamp_source != AGE_BOUND_STAMP_FROM_CONSTRAINT_ROWS:
        return (
            f"{key}={value} restates an age bound whose operator is ambiguous: "
            f"{AGE_BOUND_STAMP_SOURCE_KEY}={stamp_source} does not attest it "
            "was stamped from a constraint row (>= or <), and a dimension of "
            "that name carries no operator"
        )
    return _restated_band_edge_refusal(
        key, value, metadata, compiled_key=RESTATED_AGE_BAND_COMPILED_KEYS[side]
    )


def _restated_student_loan_balance_band_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side is None:
        return (
            f"{key}={value} restates an exact federal student loan balance, but "
            "the materializer slices a half-open balance band"
        )
    materializer = metadata.get("materializer")
    if materializer != FSA_MATERIALIZER:
        return (
            f"{key}={value} restates a federal student loan balance bound on a "
            f"spec whose materializer ({materializer!r}) applies no balance band"
        )
    return _restated_band_edge_refusal(
        key,
        value,
        metadata,
        compiled_key=RESTATED_STUDENT_LOAN_BALANCE_BAND_COMPILED_KEYS[side],
    )


def _restated_eitc_child_count_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side in RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES:
        return (
            f"{key}={value} restates a qualifying-child {side} bound, refused "
            "outright: the Ledger's operator for it (< or <=) is unconfirmed, "
            "and the two readings select different returns"
        )
    compiled = _soi_eitc_child_count_filter(metadata)
    if compiled is None:
        return (
            f"{key}={value} restates a qualifying-child bound on a spec that "
            "carries no child-count filter for the materializer to apply"
        )
    bound = _restated_count_value(value)
    if bound is None:
        return (
            f"{key}={value} is not a qualifying-child count in "
            f"0..{RESTATED_EITC_CHILD_COUNT_PROBE_MAX}"
        )
    counts = np.arange(RESTATED_EITC_CHILD_COUNT_PROBE_MAX + 1, dtype=np.float64)
    if side == "lower":
        restated_mask = counts >= bound
    elif side is None:
        restated_mask = counts == bound
    else:
        raise ValueError(
            f"{key}: bound side {side!r} reached the qualifying-child "
            "comparison, which reads only lower and exact restatements"
        )
    try:
        applied_mask = _eitc_child_count_mask(counts, compiled)
    except ValueError:
        return (
            f"{key}={value} cannot be compared: compiled child-count filter "
            f"{compiled!r} is not one the materializer understands"
        )
    if not np.array_equal(restated_mask, applied_mask):
        return (
            f"{key}={value} selects different returns than the compiled "
            f"child-count filter {compiled!r}"
        )
    return None


def _restated_ledger_filter_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
) -> str | None:
    """Judge one otherwise-unsupported ``ledger_filter_*`` key for one spec.

    Returns ``None`` only when *key* restates, in the labelled feed's concept
    vocabulary, a constraint the materializer already applies to this spec,
    and the two select the same population — ignoring that restatement then
    changes nothing. Every other key returns the entry to refuse with: the
    bare key for a filter the materializer does not model (unchanged from
    before this rule existed), or the key with both values when a restatement
    disagrees with its compiled counterpart, or has none, or is applied by
    no materializer for this spec, or — for a
    qualifying-child upper bound, whose operator is unconfirmed (see
    :data:`RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES`) — the key with that
    reason, whatever the spec compiles.

    Two compiled counterparts — the AGI band in
    :data:`RESTATED_AGI_BAND_COMPILED_KEYS` and
    :func:`_soi_eitc_child_count_filter` — are the ``irs_soi`` slice's, read
    in the loop :func:`_materialize_target_frame` runs over ``irs_soi``
    specs, and this rule reads them off metadata without consulting the
    spec's family. That family-blindness is the guard's existing shape:
    ``ledger_filter_eitc_child_count`` is likewise a blanket supported key. A
    spec of another family carries neither counterpart, so it refuses on the
    "does not compile" arm rather than being accepted by accident.

    The third, the person-age band in :data:`RESTATED_AGE_BAND_COMPILED_KEYS`,
    is read by two materializers (``population_age`` and the age-banded
    ``policyengine_variable`` branch), so that rule is family-blind too but
    names the materializers that apply the band
    (:data:`RESTATED_AGE_BAND_MATERIALIZERS`) and refuses on every other; it
    also refuses a bound on a fact whose dimensions include ``age``
    (:data:`RESTATED_AGE_DIMENSION_KEY`), and any bound the compile does not
    attest was stamped from a ``>=`` / ``<`` constraint row
    (:data:`AGE_BOUND_STAMP_SOURCE_KEY`): a dimension-stamped bound carries no
    operator, and ``age < 10`` and ``age <= 10`` select different people.
    """

    concept, side = _restated_ledger_filter_concept(key)
    rule = RESTATED_LEDGER_FILTER_CONCEPTS.get(concept) if concept else None
    if rule is None:
        return key
    if rule == "agi_band":
        return _restated_agi_band_refusal(key, value, metadata, side=side)
    if rule == "eitc_child_count":
        return _restated_eitc_child_count_refusal(key, value, metadata, side=side)
    if rule == "age_band":
        return _restated_age_band_refusal(key, value, metadata, side=side)
    if rule == "student_loan_balance_band":
        return _restated_student_loan_balance_band_refusal(
            key, value, metadata, side=side
        )
    raise ValueError(
        f"RESTATED_LEDGER_FILTER_CONCEPTS maps {concept!r} to rule {rule!r}, "
        "which has no comparison implemented."
    )
