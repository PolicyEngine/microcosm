"""Explicit source-report Social Security component basis.

These are numerical owners, not source admission APIs. A graph host must bind
the live source projection and any actual fitted completion artifact before
using the returned values. Missing components never become zero. Report-record totals are not individual
beneficiary incidence. This module neither replaces totals with PUF draws nor
allocates combined family payments to individual beneficiaries.
"""

from __future__ import annotations

import numpy as np

COMPONENTS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_dependents",
    "social_security_survivors",
)
PROTOCOL = "microcosm.us.survey-social-security-basis.v1"
REASON_COMPONENTS = {
    1: (0,),
    2: (1,),
    3: (3,),
    4: (2,),
    5: (3,),
    6: (2,),
    # The published code mixes surviving, dependent and disabled children.
    7: (1, 2, 3),
    8: (0, 1, 2, 3),
}


def require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_SOCIAL_SECURITY_" + reason)


def _float64(values, *, shape, nullable=False):
    require(type(values) is np.ndarray, "ARRAY_TYPE")
    require(values.dtype == np.dtype("float64") and values.shape == shape, "ARRAY_AXIS")
    require(not np.isinf(values).any(), "NONFINITE")
    require(nullable or not np.isnan(values).any(), "UNKNOWN")
    require((values[~np.isnan(values)] >= 0).all(), "NEGATIVE")
    return values.copy()


def asec_reason_basis(total, reason_1, reason_2):
    """Map a source total and reasons without inventing component amounts.

    Callers qualify the actual money and literal reason observations. Codes
    3/5 share survivors and 4/6 share dependents. Multiple distinct components,
    code 7, code 8, or missing reasons leave the positive amount unallocated.
    A single component receives the total as an explicit mapping judgment.
    No age fallback or priority between reported reasons is applied.
    """
    require(type(total) is np.ndarray and total.ndim == 1, "TOTAL_AXIS")
    amount = _float64(total, shape=total.shape)
    reasons = []
    for values in (reason_1, reason_2):
        arr = _float64(values, shape=total.shape, nullable=True)
        require(
            (
                (arr[np.isfinite(arr)] == np.floor(arr[np.isfinite(arr)]))
                & (arr[np.isfinite(arr)] <= 8)
            ).all(),
            "REASON_DOMAIN",
        )
        reasons.append(arr)
    out = np.full((len(amount), 4), np.nan, dtype=np.float64)
    allowed = np.ones((len(amount), 4), dtype=np.bool_)
    labels = []
    for i, value in enumerate(amount):
        if value == 0:
            out[i] = 0
            allowed[i] = False
            labels.append("known_total_zero")
            continue
        codes = [arr[i] for arr in reasons]
        if any(np.isnan(code) for code in codes):
            labels.append("unresolved_missing_reason")
            continue
        nonzero = [int(code) for code in codes if code != 0]
        if not nonzero:
            labels.append("unresolved_niu_reason_with_positive_total")
            continue
        possible = set().union(*(REASON_COMPONENTS[code] for code in nonzero))
        allowed[i] = False
        allowed[i, list(possible)] = True
        if len(possible) == 1:
            out[i] = 0
            out[i, next(iter(possible))] = value
            labels.append("source_reason_total_allocation")
        else:
            labels.append("unresolved_component_split")
    return out, allowed, tuple(labels)


def asec_reporting_basis(total, age, recipiency, reason_1, reason_2):
    """Respect the age-15 reporting universe without inferring beneficiaries.

    SS_VAL=0 outside the question universe is a source sentinel, not observed
    absence of benefits. Retain that literal upstream and return unknown total
    and shares here. A reporting adult may receive combined family payments;
    the returned component basis remains at that source report's grain.
    """
    require(type(total) is np.ndarray and total.ndim == 1, "TOTAL_AXIS")
    amount = _float64(total, shape=total.shape)
    ages = _float64(age, shape=total.shape)
    receipt = _float64(recipiency, shape=total.shape, nullable=True)
    require(((ages == np.floor(ages)) & (ages <= 99)).all(), "AGE_DOMAIN")
    require(
        np.isin(receipt[np.isfinite(receipt)], [0, 1, 2]).all(), "RECIPIENCY_DOMAIN"
    )
    eligible = ages >= 15
    require(
        ((amount[~eligible] == 0) & (receipt[~eligible] == 0)).all(),
        "OUTSIDE_REPORTING_UNIVERSE_OBSERVATION",
    )
    require(np.isin(receipt[eligible], [1, 2]).all(), "IN_UNIVERSE_RECIPIENCY_UNKNOWN")
    require(not ((amount > 0) & (receipt != 1)).any(), "RECIPIENCY_CONTRADICTION")
    basis, allowed, labels = asec_reason_basis(amount, reason_1, reason_2)
    amount[~eligible] = np.nan
    basis[~eligible] = np.nan
    allowed[~eligible] = True
    labels = tuple(
        label if inside else "asec_below15_outside_reporting_universe"
        for label, inside in zip(labels, eligible, strict=True)
    )
    return amount, basis, allowed, labels


def complete_positive_basis(total, source_basis, allowed, modeled_scores):
    """Convert actual model scores to shares only for unresolved positive rows.

    The graph owner must authenticate the fitted artifact. This numerical API
    cannot do so. Existing complete rows must have no modeled scores; eligible
    rows require a complete nonnegative four-vector with positive allowed mass.
    Source totals and excluded component possibilities are retained exactly.
    A missing/zero model vector refuses instead of creating equal shares.
    """
    require(type(total) is np.ndarray and total.ndim == 1, "TOTAL_AXIS")
    amount = _float64(total, shape=total.shape)
    shape = (len(amount), 4)
    basis = _float64(source_basis, shape=shape, nullable=True)
    scores = _float64(modeled_scores, shape=shape, nullable=True)
    require(
        type(allowed) is np.ndarray
        and allowed.dtype == np.dtype("bool")
        and allowed.shape == shape,
        "ALLOWED_AXIS",
    )
    complete = np.isfinite(basis).all(axis=1)
    require((complete | np.isnan(basis).all(axis=1)).all(), "PARTIAL_COMPONENT_BASIS")
    require(np.isnan(scores[complete]).all(), "MODELED_SOURCE_OVERWRITE")
    require(
        np.array_equal(basis[complete].sum(axis=1), amount[complete]),
        "SOURCE_TOTAL_IDENTITY",
    )
    require((basis[complete][~allowed[complete]] == 0).all(), "SOURCE_SUPPORT")
    missing = ~complete
    require((amount[missing] > 0).all(), "UNRESOLVED_ZERO_TOTAL")
    require(np.isfinite(scores[missing]).all(), "COMPLETION_UNKNOWN")
    selected = scores[missing].copy()
    selected[~allowed[missing]] = 0
    mass = selected.sum(axis=1)
    require(np.isfinite(mass).all() and (mass > 0).all(), "COMPLETION_NO_ALLOWED_MASS")
    filled = selected / mass[:, None] * amount[missing, None]
    # Put the floating residual on the largest supported component. This keeps
    # excluded components exactly zero and avoids systematic lost pennies.
    if len(filled):
        j = filled.argmax(axis=1)
        filled[np.arange(len(filled)), j] += amount[missing] - filled.sum(axis=1)
    require(np.isfinite(filled).all() and (filled >= 0).all(), "COMPLETION_ARITHMETIC")
    basis[missing] = filled
    return basis
