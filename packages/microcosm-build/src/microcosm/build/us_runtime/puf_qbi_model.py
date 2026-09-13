"""Explicit canonical PUF QBI data model, adapted from the archived PUF owner.

These modeled qualification and business-input leaves are not raw IRS facts
or tax deductions. PolicyEngine owns the statutory calculation. Source cells
must be complete before this boundary; no missing cell becomes an assumed zero.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .operator_column_contracts import US_QBI_OUTPUT_COLUMNS

VERSION = "microcosm.us.puf_canonical_qbi_model/1"
ARCHIVED_COMMIT = "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe"
ARCHIVED_CODE_SHA256 = (
    "359fd71242315e5abb7b857bb83d079ac62e46b3edc46adf51481efaa6ac858e"
)
ARCHIVED_ASSUMPTIONS_SHA256 = (
    "b32f29c07061ddf41dae12f8614e033347122aa05943225dcc6bdb2bdfbb48b7"
)
SOURCES = (
    "self_employment_income",
    "farm_operations_income",
    "farm_rent_income",
    "rental_income",
    "estate_income",
    "partnership_s_corp_income",
)
REQUIRED_INPUTS = (
    "self_employment_income_before_lsr",
    "farm_operations_income",
    "farm_rent_income",
    "rental_income",
    "estate_income",
    "partnership_income",
    "s_corp_income",
    "non_qualified_dividend_income",
)
OUTPUTS = (*US_QBI_OUTPUT_COLUMNS, "self_employment_income_before_lsr")
# Exact source-ordered values from archived_qbi_assumptions.yaml. The source
# archive hash is provenance; executable parameter identity hashes these values.
_PARAMETERS_JSON = """{
 "qualification": [0.95,0.98,0.80,0.70,0.60,0.90],
 "sstb_sources": ["self_employment_income","partnership_s_corp_income","estate_income"],
 "sstb_probabilities": [0.30,0.25,0.15],
 "profit_margin": [[2,2,0.50,0.05],[2,4,0.45,0.03],[2,3,0.45,0.05],[2.5,3,0.50,0.05],[2,3,0.45,0.05],[2,3.5,0.45,0.05]],
 "employee_logit": {"slope_per_dollar":1.2e-6,"target_share":0.18},
 "labor_ratio": [[2,2,0.25,0],[2,2.5,0.22,0],[1.5,8,0.08,0],[1.5,8,0.08,0],[2,2.5,0.20,0],[2,2,0.22,0]],
 "ubia": {"sigma":1.0,"multiple":[1.5,4,8,10,3,3],"capital_probability":[0.25,0.70,0.90,0.95,0.50,0.45]},
 "reit_ptp": [["non_qualified_dividend_income",0.35,2,18,1],["partnership_s_corp_income",0.05,2,8,0.60]],
 "bdc": [["non_qualified_dividend_income",0.05,2,18,0.30]]
}"""


def model_parameters():
    """Return a fresh copy, avoiding mutable global assumptions."""
    return json.loads(_PARAMETERS_JSON)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


PARAMETERS_SHA256 = _sha(_json(model_parameters()).encode())


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _vector(value, name, n):
    values = np.asarray(value)
    _require(
        values.ndim == 1 and len(values) == n and values.dtype.kind in "ifu",
        "QBI_PHYSICAL_TYPE:" + name,
    )
    if values.dtype.kind in "iu":
        _require(
            bool(((values >= -(2**53)) & (values <= 2**53)).all()),
            "QBI_INTEGER_PRECISION:" + name,
        )
    values = values.astype(np.float64)
    _require(bool(np.isfinite(values).all()), "QBI_NONFINITE:" + name)
    return values


def _ids(recids):
    values = np.asarray(recids)
    _require(
        values.ndim == 1
        and len(values) > 0
        and values.dtype.kind in "iu"
        and bool(((values > 0) & (values <= np.iinfo(np.int64).max)).all()),
        "QBI_RECID_PHYSICAL_TYPE",
    )
    _require(len(np.unique(values)) == len(values), "QBI_DUPLICATE_RECID")
    return values.astype("<i8")


def _digest_columns(ids, columns):
    order = np.argsort(ids)
    digest = hashlib.sha256(ids[order].astype("<i8").tobytes())
    for name, values in columns.items():
        digest.update(name.encode() + b"\x00")
        digest.update(np.asarray(values[order], dtype="<f8").tobytes())
    return digest.hexdigest()


def _logistic(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700, 700)))


def _employee_intercept(revenues, slope, target):
    positive = revenues > 0
    if not positive.any():
        return 0.0
    term = slope * revenues[positive]
    target_logit = np.log(target / (1 - target))
    low, high = target_logit - np.max(term) - 80, target_logit - np.min(term) + 80
    for _ in range(100):
        mid = (low + high) / 2
        if _logistic(mid + term).mean() < target:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)


@dataclass(frozen=True)
class QbiEmploymentCalibration:
    """A reusable statistical fit; this numerical record is not source authority."""

    version: str
    parameters_sha256: str
    seed: int
    fitted_cohort_sha256: str
    fitted_rows: int
    positive_receipt_rows: int
    intercept: float


@dataclass(frozen=True)
class QbiModelResult:
    columns: Mapping[str, np.ndarray]
    calibration: QbiEmploymentCalibration
    receipt: Mapping[str, object]


def model_full_puf_qbi(
    canonical, recids, *, known, input_money_year, seed=0, employment_calibration=None
):
    """Model 15 QBI leaves and explicitly replace the ordinary Schedule C leaf.

    Knownness here means complete canonical input, possibly derived upstream;
    it does not mean a modeled leaf was observed. Inputs and their RECIDs must
    already belong to the ordinary-return cohort selected by the source owner.
    Each RNG is keyed by RECID, seed, and the archived stream number. This is
    intentionally not byte parity with the archive's row-position RNG. The
    employee logit intercept is fit to the unweighted positive-receipt cohort
    (the archive's 18% target); reuse the returned fit for stable subset/chunk
    application. Fitting a different cohort is a different statistical model.
    """
    ids = _ids(recids)
    n = len(ids)
    _require(
        type(input_money_year) is int and input_money_year == 2015,
        "QBI_INPUT_MONEY_YEAR",
    )
    _require(type(seed) is int and 0 <= seed < 2**64, "QBI_SEED")
    _require(
        isinstance(canonical, Mapping) and isinstance(known, Mapping),
        "QBI_INPUT_MAPPING",
    )
    inputs = {}
    for name in REQUIRED_INPUTS:
        _require(name in canonical and name in known, "QBI_MISSING_INPUT:" + name)
        mask = np.asarray(known[name])
        _require(
            mask.shape == (n,) and mask.dtype.kind == "b", "QBI_KNOWNNESS_TYPE:" + name
        )
        _require(bool(mask.all()), "QBI_UNKNOWN:" + name)
        inputs[name] = _vector(canonical[name], name, n)
    source = np.column_stack(
        [inputs[name] for name in REQUIRED_INPUTS[:5]]
        + [inputs["partnership_income"] + inputs["s_corp_income"]]
    )
    _require(bool(np.isfinite(source).all()), "QBI_SOURCE_SUM_NONFINITE")
    params = model_parameters()
    qualified = np.zeros((n, 6), dtype=bool)
    margins, labor = np.zeros((n, 6)), np.zeros((n, 6))
    employees_draw, capital_draw, normal_draw, sstb_draw = np.zeros((4, n))
    investment_draws = np.zeros((n, 3))
    # Generate each record independently; calculations below remain vectorized.
    for row, recid in enumerate(ids):

        def rng(stream, recid=int(recid)):
            return np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([seed, stream, recid]))
            )

        qualified[row] = rng(41).random(6) < params["qualification"]
        business = rng(42)
        for col, (a, b, scale, shift) in enumerate(params["profit_margin"]):
            margins[row, col] = business.beta(a, b) * scale + shift
        employees_draw[row] = business.random()
        for col, (a, b, scale, shift) in enumerate(params["labor_ratio"]):
            labor[row, col] = business.beta(a, b) * scale + shift
        capital_draw[row] = business.random()
        normal_draw[row] = business.standard_normal()
        sstb_draw[row] = rng(64).random()
        investment = rng(43)
        for col, (_name, probability, a, b, scale) in enumerate(
            params["reit_ptp"] + params["bdc"]
        ):
            receives = investment.random() < probability
            share = np.clip(investment.beta(a, b) * scale, 0, 1)
            investment_draws[row, col] = share if receives else 0.0
    components = source * qualified
    positive = np.maximum(components, 0)
    positive_total = positive.sum(axis=1)
    _require(bool(np.isfinite(positive_total).all()), "QBI_POSITIVE_TOTAL_NONFINITE")
    fractions = np.divide(
        positive,
        positive_total[:, None],
        out=np.zeros_like(positive),
        where=positive_total[:, None] > 0,
    )
    qbi = components.sum(axis=1)
    _require(bool(np.isfinite(qbi).all()), "QBI_NET_TOTAL_NONFINITE")
    margin = (fractions * margins).sum(axis=1)
    revenues = np.divide(np.maximum(qbi, 0), margin, out=np.zeros(n), where=margin > 0)
    _require(bool(np.isfinite(revenues).all()), "QBI_REVENUE_NONFINITE")
    cohort_sha = _digest_columns(ids, inputs)
    logit = params["employee_logit"]
    if employment_calibration is None:
        # Sort by identity before reduction so input permutations replay exactly.
        employment_calibration = QbiEmploymentCalibration(
            VERSION,
            PARAMETERS_SHA256,
            seed,
            cohort_sha,
            n,
            int((revenues > 0).sum()),
            _employee_intercept(
                revenues[np.argsort(ids)],
                logit["slope_per_dollar"],
                logit["target_share"],
            ),
        )
    _require(
        isinstance(employment_calibration, QbiEmploymentCalibration)
        and employment_calibration.version == VERSION
        and employment_calibration.parameters_sha256 == PARAMETERS_SHA256
        and employment_calibration.seed == seed
        and type(employment_calibration.intercept) is float
        and np.isfinite(employment_calibration.intercept),
        "QBI_EMPLOYMENT_CALIBRATION_IDENTITY",
    )
    probability = np.where(
        revenues > 0,
        _logistic(
            employment_calibration.intercept + logit["slope_per_dollar"] * revenues
        ),
        0.0,
    )
    wages = revenues * (fractions * labor).sum(axis=1) * (employees_draw < probability)
    capital_probability = fractions @ params["ubia"]["capital_probability"]
    mean_ubia = (fractions @ params["ubia"]["multiple"]) * np.maximum(qbi, 0)
    ubia = np.where(
        (capital_draw < capital_probability) & (mean_ubia > 0),
        mean_ubia
        * np.exp(
            params["ubia"]["sigma"] * normal_draw - params["ubia"]["sigma"] ** 2 / 2
        ),
        0.0,
    )
    # Canonical source names ensure the SSTB draw consumes qualification flags.
    sstb_positions = [SOURCES.index(name) for name in params["sstb_sources"]]
    sstb_sources = positive[:, sstb_positions]
    largest = sstb_sources.argmax(axis=1)
    sstb_probability = np.where(
        sstb_sources.sum(axis=1) > 0,
        np.asarray(params["sstb_probabilities"])[largest],
        0.0,
    )
    is_sstb = sstb_draw < sstb_probability
    output = {
        name + "_would_be_qualified": qualified[:, i].copy()
        for i, name in enumerate(SOURCES)
    }
    schedule_c = inputs["self_employment_income_before_lsr"]
    output.update(
        {
            "business_is_sstb": is_sstb,
            "self_employment_income_before_lsr": np.where(is_sstb, 0, schedule_c),
            "sstb_self_employment_income_before_lsr": np.where(is_sstb, schedule_c, 0),
            "self_employment_income_would_be_qualified": qualified[:, 0] & ~is_sstb,
            "sstb_self_employment_income_would_be_qualified": qualified[:, 0] & is_sstb,
            "w2_wages_from_qualified_business": wages,
            "unadjusted_basis_qualified_property": ubia,
            "sstb_w2_wages_from_qualified_business": np.where(is_sstb, wages, 0),
            "sstb_unadjusted_basis_qualified_property": np.where(is_sstb, ubia, 0),
            "qualified_reit_and_ptp_income": np.maximum(
                inputs["non_qualified_dividend_income"], 0
            )
            * investment_draws[:, 0]
            + np.maximum(source[:, 5], 0) * investment_draws[:, 1],
            "qualified_bdc_income": np.maximum(
                inputs["non_qualified_dividend_income"], 0
            )
            * investment_draws[:, 2],
        }
    )
    _require(set(output) == set(OUTPUTS), "QBI_OUTPUT_ROSTER")
    for name, values in output.items():
        _require(bool(np.isfinite(values).all()), "QBI_OUTPUT_NONFINITE:" + name)
    frozen = {
        name: np.frombuffer(
            np.asarray(output[name]).tobytes(), dtype=output[name].dtype
        )
        for name in OUTPUTS
    }
    receipt = {
        "version": VERSION,
        "origin": "modeled_not_observed",
        "release_eligible": False,
        "archived_commit": ARCHIVED_COMMIT,
        "archived_code_sha256": ARCHIVED_CODE_SHA256,
        "archived_assumptions_sha256": ARCHIVED_ASSUMPTIONS_SHA256,
        "parameters_sha256": PARAMETERS_SHA256,
        "input_money_year": input_money_year,
        "input_money_unit": "nominal USD",
        "source_statistical_baseline": 2015,
        "growth": "not applied; employee slope per dollar is bound to this basis",
        "sources": list(REQUIRED_INPUTS),
        "input_cohort_sha256": cohort_sha,
        "rows": n,
        "seed": seed,
        "randomness": "PCG64 SeedSequence([seed, archived_stream, RECID]); streams 41/42/43/64",
        "numpy_version": np.__version__,
        "input_knownness": "complete_canonical_not_raw_observation",
        "employment_fit": dict(employment_calibration.__dict__),
        "applied_positive_receipt_rows": int((revenues > 0).sum()),
        "applied_employee_expected_share": float(probability[revenues > 0].mean())
        if (revenues > 0).any()
        else 0.0,
        "employee_target_weighting": "unweighted positive simulated receipts, archived target 0.18",
        "subset_replay": "reuse employment_calibration; refitting a subset changes the model",
        "schedule_c": "all-or-nothing SSTB split preserving the signed input total",
        "wages_and_ubia": "base leaves are total pools; SSTB leaves are conditional copies",
        "output_columns": list(OUTPUTS),
        "output_values_sha256": _digest_columns(ids, frozen),
    }
    receipt["sha256"] = _sha(_json(receipt).encode())
    return QbiModelResult(
        MappingProxyType(frozen), employment_calibration, MappingProxyType(receipt)
    )
