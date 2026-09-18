"""Return-grain canonical PUF59 with explicit model/growth lineage.

No tax engine, invented donor people, loan balances or loan vintages. The
observed statistical2015 source remains immutable beside modeled outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import puf_full_source as source_owner
from . import puf_qbi_model as qbi
from . import puf_target2024_growth as growth

VERSION = "microcosm.us.puf59_canonical_return/2"
INTEREST_ASSET_SHA256 = (
    "c3356ae216487f365cb0e0a7ab1ba46843a6c52950b75eae1bfab9b0b80a735a"
)

INTEREST_BAND_FACTS_SHA256 = (
    "3ea894c474302dd818f347f5bfde3b331bf80468b0ac928ffe76ed1e6de62cbc"
)
PREFIX_NAMES = (
    "RECID",
    "weight",
    "puf_person_incidence_capacity",
    "puf_2015_filing_status_code",
    "puf_2015_capped_return_size",
)


def prefix_values_digest(arrays):
    """Bind identities, source weights and source-specific features in row order."""
    digest = hashlib.sha256()
    for name in PREFIX_NAMES:
        digest.update(name.encode("ascii") + b"\0")
        digest.update(
            np.asarray(
                arrays[name], dtype="<f8" if name == "weight" else "<i8"
            ).tobytes()
        )
    return digest.hexdigest()


def _require(value, code):
    if not value:
        raise ValueError(code)


def _money(value, name, n):
    a = np.asarray(value)
    _require(
        a.shape == (n,) and a.dtype.kind in "ifu", "PUF59_MODEL_PHYSICAL_TYPE:" + name
    )
    if a.dtype.kind in "iu":
        _require(
            bool(((a >= -(2**53)) & (a <= 2**53)).all()),
            "PUF59_MODEL_INTEGER_PRECISION:" + name,
        )
    a = a.astype(np.float64)
    _require(bool(np.isfinite(a).all()), "PUF59_MODEL_NONFINITE:" + name)
    return a


def baseline_modeled_columns(auxiliary, *, n, interest_bands, interest_asset_sha256):
    """Eleven explicit baseline outputs absent as exact raw canonical leaves."""
    _require(
        interest_asset_sha256 == INTEREST_ASSET_SHA256, "PUF59_INTEREST_ASSET_IDENTITY"
    )
    _require(isinstance(auxiliary, Mapping), "PUF59_MODEL_AUXILIARY_MAPPING")
    names = (
        "raw_adjusted_gross_income",
        "raw_total_interest_deduction",
        "raw_total_social_security",
        "raw_realized_ira_deduction",
        "raw_realized_keogh_deduction",
        "raw_tuition_fees_deduction",
        "raw_lifetime_learning_qualified_expenses",
        "raw_miscellaneous_itemized_deductions",
        "raw_partnership_nonpassive_net",
    )
    _require(all(k in auxiliary for k in names), "PUF59_MODEL_MISSING_AUXILIARY")
    a = {k: _money(auxiliary[k], k, n) for k in names}
    for k in names:
        if k not in ("raw_adjusted_gross_income", "raw_partnership_nonpassive_net"):
            _require(bool((a[k] >= 0).all()), "PUF59_MODEL_NEGATIVE_SOURCE:" + k)
    bands = tuple(interest_bands)
    _require(len(bands) == 22, "PUF59_INTEREST_BAND_COUNT")
    _require(
        bands[0].lower_bound is None and bands[-1].upper_bound is None,
        "PUF59_INTEREST_FULL_AGI",
    )
    for left, right in zip(bands[:-1], bands[1:], strict=True):
        _require(left.upper_bound == right.lower_bound, "PUF59_INTEREST_CONTIGUOUS")
    total = a["raw_total_interest_deduction"]
    agi = a["raw_adjusted_gross_income"]
    investment = np.zeros(n)
    covered = np.zeros(n, dtype=np.uint8)
    band_facts = []
    fields = (
        "home_mortgage_interest_amount",
        "deductible_points_amount",
        "qualified_mortgage_insurance_premiums_amount",
        "investment_interest_amount",
    )
    for i, band in enumerate(bands):
        _require(band.source_row == 11 + i, "PUF59_INTEREST_SOURCE_ROW")
        lo, hi = band.lower_bound, band.upper_bound
        _require(
            (lo is None or type(lo) in (int, float) and np.isfinite(lo))
            and (hi is None or type(hi) in (int, float) and np.isfinite(hi))
            and (lo is None or hi is None or lo < hi),
            "PUF59_INTEREST_BAND_BOUNDS",
        )
        amounts = np.asarray([getattr(band, k) for k in fields], dtype=np.float64)
        _require(
            bool(np.isfinite(amounts).all() and (amounts >= 0).all())
            and amounts.sum() > 0,
            "PUF59_INTEREST_COMPONENT_DOMAIN",
        )
        _require(
            type(band.total_interest_paid_amount) is int
            and abs(
                band.total_interest_paid_amount - sum(getattr(band, k) for k in fields)
            )
            <= 1,
            "PUF59_INTEREST_SOURCE_ROUNDING",
        )
        mask = np.ones(n, dtype=bool)
        if lo is not None:
            mask &= agi >= lo
        if hi is not None:
            mask &= agi < hi
        investment[mask] = total[mask] * (amounts[3] / amounts.sum())
        covered[mask] += 1
        band_facts.append(
            {
                "source_row": band.source_row,
                "lower_bound": lo,
                "upper_bound": hi,
                "amounts_thousands_usd": amounts.tolist(),
            }
        )
    band_facts_sha256 = hashlib.sha256(
        json.dumps(band_facts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _require(
        band_facts_sha256 == INTEREST_BAND_FACTS_SHA256,
        "PUF59_INTEREST_BAND_FACTS_IDENTITY",
    )
    _require(bool((covered == 1).all()), "PUF59_INTEREST_PARTITION")
    home = total - investment
    investment = total - home
    _require(np.array_equal(home + investment, total), "PUF59_INTEREST_CONSERVATION")
    columns = {
        "home_mortgage_interest": home,
        "investment_interest_expense": investment,
        "social_security_retirement": a["raw_total_social_security"].copy(),
        "social_security_disability": np.zeros(n),
        "social_security_dependents": np.zeros(n),
        "social_security_survivors": np.zeros(n),
        "traditional_ira_contributions_desired": a["raw_realized_ira_deduction"].copy(),
        "self_employed_pension_contributions_desired": a[
            "raw_realized_keogh_deduction"
        ].copy(),
        "qualified_tuition_expenses": np.maximum(
            a["raw_tuition_fees_deduction"],
            a["raw_lifetime_learning_qualified_expenses"],
        ),
        "unreimbursed_business_employee_expenses": a[
            "raw_miscellaneous_itemized_deductions"
        ].copy(),
        "partnership_self_employment_net_earnings": a[
            "raw_partnership_nonpassive_net"
        ].copy(),
    }
    return columns, {
        "schema": "microcosm.us.puf59_baseline_models/1",
        "input_money_year": 2015,
        "interest_asset_sha256": interest_asset_sha256,
        "interest_band_facts_sha256": band_facts_sha256,
        "interest_method": "Normalize all four published component amounts. Map mortgage+points+insurance to the available home interest leaf and investment only to investment expense; complementary rounding conserves E19200 exactly.",
        "ss_method": "Component-neutral carrier: total SS in retirement, other component carriers zero; requires recipient-profile SS reconciliation. Zeros are not observations of absent benefits.",
        "desired_deductions": "Realized IRA/Keogh deductions proxy for desired amounts; uncensored demand is not observed.",
        "tuition": "Maximum of realized tuition deduction and Lifetime Learning qualified expenses; incomplete source expense proxy.",
        "employee_expense": "Total miscellaneous itemized deductions proxy; broader than employee business expenses.",
        "partnership_earnings": "Nonpassive partnership net earnings proxy; not reconstructed from capped ScheduleSE amounts.",
        "mortgage_structure_generated": False,
        "origin": "modeled_not_observed",
    }


@dataclass(frozen=True)
class CanonicalPuf59Result:
    columns: Mapping[str, np.ndarray]
    source_predictors: Mapping[str, np.ndarray]
    recids: np.ndarray
    design_weight: np.ndarray
    person_incidence_capacity: np.ndarray
    money_year: int
    qbi_calibration: qbi.QbiEmploymentCalibration
    receipt: Mapping[str, object]


def construct_canonical_puf59(
    source,
    *,
    interest_bands,
    interest_asset_sha256,
    selected_recids=None,
    qbi_employment_calibration=None,
    growth_scheme="family_observed",
    seed=0,
):
    """Construct canonical outputs from an owner-decoded ordinary source cohort.

    The caller supplies source-authenticated interest bands. Selection is owned
    by the source decoder. No data source is opened or sampled in this function.
    """
    observed, aux, status = source_owner.observed_and_derived_return_columns(
        source, selected_recids=selected_recids
    )
    n = len(status["RECID"])
    _require(n > 0, "PUF59_EMPTY_COHORT")
    modeled, model_receipt = baseline_modeled_columns(
        aux,
        n=n,
        interest_bands=interest_bands,
        interest_asset_sha256=interest_asset_sha256,
    )
    _require(not set(observed) & set(modeled), "PUF59_CANONICAL_COLLISION")
    canonical = {**observed, **modeled}
    known = {k: np.ones(n, dtype=bool) for k in canonical}
    qresult = qbi.model_full_puf_qbi(
        canonical,
        status["RECID"],
        known=known,
        input_money_year=2015,
        seed=seed,
        employment_calibration=qbi_employment_calibration,
    )
    canonical.update(qresult.columns)
    _require(set(canonical) == set(growth.OUTPUTS), "PUF59_CANONICAL_COMPLETE_ROSTER")
    known = {k: np.ones(n, dtype=bool) for k in canonical}
    result = growth.grow_puf_2015_to_2024(
        canonical, known=known, input_money_year=2015, scheme=growth_scheme
    )
    weight = status["S006"].astype(np.float64) / 100
    _require(
        bool((weight >= 0).all() and np.isfinite(weight).all() and weight.sum() > 0),
        "PUF59_WEIGHT_DOMAIN",
    )
    predictors = {
        k: np.frombuffer(np.asarray(aux[k], dtype=np.int64).tobytes(), dtype=np.int64)
        for k in ("puf_2015_filing_status_code", "puf_2015_capped_return_size")
    }
    prefix_arrays = {
        "RECID": status["RECID"],
        "weight": weight,
        "puf_person_incidence_capacity": np.ones(n, dtype=np.int64),
        **predictors,
    }
    receipt = {
        "prefix_values_sha256": prefix_values_digest(prefix_arrays),
        "schema": VERSION,
        "source_definition_sha256": source.definition_sha256,
        "source_sha256": dict(source.source_sha256),
        "source_statistical_year": 2015,
        "input_money_year": 2015,
        "output_money_year": 2024,
        "rows": n,
        "model_assumptions": model_receipt,
        "qbi": dict(qresult.receipt),
        "growth": dict(result.receipt),
        "return_size": "Source PUF2015 coarsened filing-status plus capped dependent count; no physical person claim.",
        "person_incidence_capacity": "Constant1 for zero/one modeled return incidence; excluded from conditioning predictors.",
        "weight": "Source S006 exact hundredths divided once by100; no weight growth.",
        "recipient_conditioning_self_employment": "Must sum base and SSTB ScheduleC outcomes for the total source predictor.",
        "source_observation_status": "33 direct/derived baseline columns; 11 explicit baseline models; 15 added QBI leaves and one replacement; all monetary transport is modeled.",
        "mortgage_structure_generated": False,
        "release_eligible": False,
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()

    def readonly(a):
        return np.frombuffer(a.tobytes(), dtype=a.dtype)

    return CanonicalPuf59Result(
        result.columns,
        MappingProxyType(predictors),
        readonly(status["RECID"].astype(np.int64)),
        readonly(weight),
        readonly(np.ones(n, dtype=np.int64)),
        2024,
        qresult.calibration,
        MappingProxyType(receipt),
    )
