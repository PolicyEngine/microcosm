"""Build and validate the evidence-consuming QBI v3 assumptions resource.

The restricted PUF replay belongs to the versioned assumptions-build step,
not to the simulation.  This module reduces that replay to persisted form
weights, SCF employer probabilities, solved log-odds shifts, SOI mixtures,
profit-margin inverse-CDF knots, and diagnostic anchors.  The runtime can
therefore remain a pure function of committed resources and seeded inputs.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any

import numpy as np

from populace.build.us_runtime.qbi_v3_evidence import (
    SCF_INCOME_BANDS,
    SCF_MINIMUM_UNWEIGHTED_N,
    validate_qbi_employer_structure_resource,
    validate_qbi_wage_capital_priors_resource,
)

__all__ = [
    "QBI_V3_EMPLOYER_RESOURCE",
    "QBI_V3_FORMS",
    "QBI_V3_FORM_CODES",
    "QBI_V3_NEW_SEEDS",
    "QBI_V3_RETAINED_SEEDS",
    "QBI_V3_SCHEMA_VERSION",
    "QBI_V3_SIMULATION_VERSION",
    "QBI_V3_WAGE_CAPITAL_RESOURCE",
    "assign_qbi_v3_record_forms",
    "build_qbi_v3_assumptions_payload",
    "build_qbi_v3_employer_base_probabilities",
    "build_qbi_v3_profit_margin_curves",
    "build_qbi_v3_soi_mixtures",
    "calibrate_qbi_v3_employer_shifts",
    "validate_qbi_v3_assumptions_payload",
]

QBI_V3_SCHEMA_VERSION = 3
QBI_V3_SIMULATION_VERSION = 3
QBI_V3_ENGINE = "derived_qualification_host_sstb_evidence_wage_capital_v3"
QBI_V3_EMPLOYER_RESOURCE = "qbi_employer_structure_v1.json"
QBI_V3_WAGE_CAPITAL_RESOURCE = "qbi_wage_capital_priors_v1.json"
QBI_V3_PARENT_RESOURCE = "qbi_assumptions_v2.json"

QBI_V3_FORMS: tuple[str, ...] = (
    "sole_proprietorship",
    "partnership",
    "s_corporation",
)
QBI_V3_FORM_CODES = {form: index for index, form in enumerate(QBI_V3_FORMS)}
QBI_V3_SCF_FORM = {
    "sole_proprietorship": "sole_or_informal",
    "partnership": "partnership_or_llc",
    "s_corporation": "s_corporation",
}
QBI_V3_RETAINED_SEEDS = {
    "qualification": 2041,
    "sstb": 2064,
    "investment": 2043,
}
QBI_V3_NEW_SEEDS = {
    "entity_split": 3041,
    "latent_industry": 3042,
    "employer_gate": 3043,
    "margin_quantile": 3044,
    "ubia_dispersion": 3045,
}
QBI_V3_SEEDS = {**QBI_V3_RETAINED_SEEDS, **QBI_V3_NEW_SEEDS}

_PARTNERSHIP_ENTITY_SHARE = 0.17
_S_CORPORATION_ENTITY_SHARE = 0.53
_SOLE_PROPRIETORSHIP_ENTITY_SHARE = 0.28
_ESTATE_TRUST_ENTITY_SHARE = 0.02
_SUPPORTED_FORM_SHARE = 0.98
_PARTNERSHIP_PROBABILITY = _PARTNERSHIP_ENTITY_SHARE / (
    _PARTNERSHIP_ENTITY_SHARE + _S_CORPORATION_ENTITY_SHARE
)
_S_CORPORATION_PROBABILITY = 1.0 - _PARTNERSHIP_PROBABILITY

_OVERALL_ZERO_EMPLOYEE_TARGET = 0.842
_SOLE_ZERO_EMPLOYEE_TARGET = 0.95
_PARTNERSHIP_ZERO_EMPLOYEE_TARGET = 0.80
_SOLVER_ITERATIONS = 200
_SOLVER_LOWER_BOUND = -100.0
_SOLVER_UPPER_BOUND = 100.0
_PROBABILITY_CLIP = 1e-12
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object.")
    return value


def _list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a JSON array.")
    return value


def _exact_keys(
    value: object,
    expected: Sequence[str] | set[str] | frozenset[str],
    location: str,
) -> Mapping[str, Any]:
    result = _mapping(value, location)
    expected_set = set(expected)
    if set(result) != expected_set:
        raise ValueError(
            f"{location} keys must be exactly {sorted(expected_set)}; "
            f"got {sorted(result)}."
        )
    return result


def _number(
    value: object,
    location: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" and at least {minimum}"
        raise ValueError(f"{location} must be finite{suffix}.")
    return result


def _probability(value: object, location: str) -> float:
    result = _number(value, location)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{location} must lie in [0, 1].")
    return result


def _positive_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a nonnegative integer.")
    return value


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string.")
    return value


def _sha256(value: object, location: str) -> str:
    result = _text(value, location)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{location} must be a lowercase SHA-256 digest.")
    return result


def _logistic(value: np.ndarray | float) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


def _logit(value: np.ndarray | float) -> np.ndarray:
    values = np.clip(
        np.asarray(value, dtype=np.float64),
        _PROBABILITY_CLIP,
        1.0 - _PROBABILITY_CLIP,
    )
    return np.log(values / (1.0 - values))


def _income_band_index(values: np.ndarray) -> np.ndarray:
    """Return indices into the persisted SCF income-band order."""

    return np.select(
        (
            values <= 0.0,
            values <= 25_000.0,
            values <= 100_000.0,
            values <= 250_000.0,
            values <= 1_000_000.0,
        ),
        (0, 1, 2, 3, 4),
        default=5,
    ).astype(np.int8)


def assign_qbi_v3_record_forms(
    qualified_components: np.ndarray,
    *,
    entity_split_seed: int = QBI_V3_NEW_SEEDS["entity_split"],
) -> tuple[np.ndarray, np.ndarray]:
    """Assign one latent legal form to each positive-QBI replay record.

    A positive qualified partnership/S-corporation component takes precedence
    and is split using the JCT 17/53 deduction shares.  Every other record
    with positive qualified QBI uses the sole-proprietorship evidence proxy.
    The entity stream draws once for every input row before applying support
    masks, pinning the family stream independently of the realized support.
    """

    components = np.asarray(qualified_components, dtype=np.float64)
    if components.ndim != 2 or components.shape[1] != 6:
        raise ValueError("QBI v3 qualified components must have shape (n, 6).")
    if not np.isfinite(components).all():
        raise ValueError("QBI v3 qualified components must be finite.")
    if isinstance(entity_split_seed, bool) or not isinstance(entity_split_seed, int):
        raise ValueError("QBI v3 entity-split seed must be an integer.")
    if entity_split_seed < 0:
        raise ValueError("QBI v3 entity-split seed must be nonnegative.")

    qbi = components.sum(axis=1)
    passthrough_positive = components[:, -1] > 0.0
    entity_draw = np.random.default_rng(entity_split_seed).random(len(components))
    form_codes = np.full(len(components), -1, dtype=np.int8)
    form_codes[(qbi > 0.0) & ~passthrough_positive] = QBI_V3_FORM_CODES[
        "sole_proprietorship"
    ]
    form_codes[
        (qbi > 0.0) & passthrough_positive & (entity_draw < _PARTNERSHIP_PROBABILITY)
    ] = QBI_V3_FORM_CODES["partnership"]
    form_codes[
        (qbi > 0.0) & passthrough_positive & (entity_draw >= _PARTNERSHIP_PROBABILITY)
    ] = QBI_V3_FORM_CODES["s_corporation"]
    return qbi, form_codes


def _aggregated_scf_counts(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float]:
    effective_n = sum(
        _number(
            _mapping(cell["requested_counts"], "requested_counts")[
                "implicate_adjusted_unweighted_n"
            ],
            "requested_counts.implicate_adjusted_unweighted_n",
            minimum=0.0,
        )
        for cell in cells
    )
    total_weight = sum(
        _number(
            _mapping(cell["requested_counts"], "requested_counts")[
                "weighted_business_interests"
            ],
            "requested_counts.weighted_business_interests",
            minimum=0.0,
        )
        for cell in cells
    )
    employer_weight = sum(
        _number(
            _mapping(cell["requested_counts"], "requested_counts")[
                "weighted_employer_proxy_business_interests"
            ],
            "requested_counts.weighted_employer_proxy_business_interests",
            minimum=0.0,
        )
        for cell in cells
    )
    return effective_n, total_weight, employer_weight


def build_qbi_v3_employer_base_probabilities(
    employer_resource: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Marginalize SCF industry cells into the adjudicated income/form shape."""

    methodology = _mapping(employer_resource.get("methodology"), "methodology")
    minimum_n = _number(
        methodology.get("minimum_implicate_adjusted_unweighted_n"),
        "methodology.minimum_implicate_adjusted_unweighted_n",
        minimum=0.0,
    )
    cells = [
        _mapping(cell, f"cells[{index}]")
        for index, cell in enumerate(_list(employer_resource.get("cells"), "cells"))
    ]
    probabilities: dict[str, dict[str, float]] = {}
    source_levels: dict[str, dict[str, str]] = {}
    for form in QBI_V3_FORMS:
        scf_form = QBI_V3_SCF_FORM[form]
        form_cells = [
            cell for cell in cells if cell.get("legal_form_group") == scf_form
        ]
        if not form_cells:
            raise ValueError(f"SCF employer resource has no cells for {scf_form!r}.")
        _, form_weight, form_employer_weight = _aggregated_scf_counts(form_cells)
        if form_weight <= 0.0:
            raise ValueError(f"SCF employer form {scf_form!r} has zero weight.")
        form_probability = form_employer_weight / form_weight
        probabilities[form] = {}
        source_levels[form] = {}
        for income_band in SCF_INCOME_BANDS:
            selected = [
                cell for cell in form_cells if cell.get("income_band") == income_band
            ]
            if not selected:
                raise ValueError(
                    f"SCF employer resource has no {scf_form!r}/{income_band!r} cells."
                )
            effective_n, total_weight, employer_weight = _aggregated_scf_counts(
                selected
            )
            if effective_n >= minimum_n:
                if total_weight <= 0.0:
                    raise ValueError(
                        f"SCF employer {scf_form!r}/{income_band!r} has zero weight."
                    )
                probability = employer_weight / total_weight
                source_level = "income_form"
            else:
                probability = form_probability
                source_level = "form"
            probabilities[form][income_band] = float(probability)
            source_levels[form][income_band] = source_level
    return probabilities, source_levels


def _eligible_soi_industries(
    form_payload: Mapping[str, Any],
    *,
    form: str,
) -> list[Mapping[str, Any]]:
    industries = [
        _mapping(industry, f"forms.{form}.industries[{index}]")
        for index, industry in enumerate(
            _list(form_payload.get("industries"), f"forms.{form}.industries")
        )
    ]
    eligible: list[Mapping[str, Any]] = []
    for industry in industries:
        raw = _mapping(
            industry.get("raw_amounts_thousands"),
            f"forms.{form}.raw_amounts_thousands",
        )
        receipts = raw.get("receipts")
        if (
            industry.get("is_aggregate") is False
            and industry.get("industry_level") != "unallocable"
            and isinstance(receipts, (int, float))
            and not isinstance(receipts, bool)
            and math.isfinite(float(receipts))
            and float(receipts) > 0.0
            and industry.get("wage_share") is not None
            and industry.get("ubia_intensity") is not None
        ):
            eligible.append(industry)
    if not eligible:
        raise ValueError(f"SOI form {form!r} has no jointly usable industry rows.")
    return eligible


def _all_industry_row(
    form_payload: Mapping[str, Any],
    *,
    form: str,
) -> Mapping[str, Any]:
    industries = _list(form_payload.get("industries"), f"forms.{form}.industries")
    matches = [
        _mapping(industry, f"forms.{form}.industries")
        for industry in industries
        if isinstance(industry, Mapping) and industry.get("industry_level") == "all"
    ]
    if len(matches) != 1:
        raise ValueError(f"SOI form {form!r} must have exactly one all-industry row.")
    return matches[0]


def build_qbi_v3_soi_mixtures(
    wage_capital_resource: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    """Build receipts-weighted joint wage/UBIA mixtures and modest dispersion."""

    forms_payload = _mapping(wage_capital_resource.get("forms"), "forms")
    mixtures: dict[str, dict[str, Any]] = {}
    dispersion: dict[str, dict[str, float]] = {}
    for form in QBI_V3_FORMS:
        form_payload = _mapping(forms_payload.get(form), f"forms.{form}")
        eligible = _eligible_soi_industries(form_payload, form=form)
        receipts = np.asarray(
            [
                _number(
                    _mapping(
                        industry["raw_amounts_thousands"],
                        f"forms.{form}.raw_amounts_thousands",
                    )["receipts"],
                    f"forms.{form}.receipts",
                    minimum=0.0,
                )
                for industry in eligible
            ],
            dtype=np.float64,
        )
        total_receipts = float(receipts.sum())
        probabilities = receipts / total_receipts
        intensities = np.asarray(
            [
                _number(
                    industry["ubia_intensity"],
                    f"forms.{form}.ubia_intensity",
                    minimum=0.0,
                )
                for industry in eligible
            ],
            dtype=np.float64,
        )
        if np.any(intensities <= 0.0):
            raise ValueError(
                f"SOI form {form!r} needs positive UBIA intensities for log dispersion."
            )
        log_intensities = np.log(intensities)
        weighted_log_mean = float(np.sum(probabilities * log_intensities))
        weighted_log_sd = float(
            np.sqrt(np.sum(probabilities * (log_intensities - weighted_log_mean) ** 2))
        )
        effective_count = float(1.0 / np.sum(probabilities**2))
        sigma = float(weighted_log_sd / math.sqrt(effective_count))

        all_industry = _all_industry_row(form_payload, form=form)
        all_raw = _mapping(
            all_industry["raw_amounts_thousands"],
            f"forms.{form}.all.raw_amounts_thousands",
        )
        all_receipts = _number(
            all_raw["receipts"],
            f"forms.{form}.all.receipts",
            minimum=0.0,
        )
        components = []
        for industry, probability in zip(eligible, probabilities, strict=True):
            components.append(
                {
                    "industry_key": _text(
                        industry["industry_key"], f"forms.{form}.industry_key"
                    ),
                    "published_label": _text(
                        industry["published_label"],
                        f"forms.{form}.published_label",
                    ),
                    "probability": float(probability),
                    "wage_share": _number(
                        industry["wage_share"],
                        f"forms.{form}.wage_share",
                        minimum=0.0,
                    ),
                    "ubia_intensity": _number(
                        industry["ubia_intensity"],
                        f"forms.{form}.ubia_intensity",
                        minimum=0.0,
                    ),
                    "proxy": industry["proxy"],
                    "capital_measure": _text(
                        industry["capital_measure"],
                        f"forms.{form}.capital_measure",
                    ),
                }
            )
        proxy_values = {component["proxy"] for component in components}
        capital_measures = {component["capital_measure"] for component in components}
        if len(proxy_values) != 1 or len(capital_measures) != 1:
            raise ValueError(f"SOI form {form!r} has inconsistent capital metadata.")
        mixtures[form] = {
            "tax_year": form_payload["tax_year"],
            "scf_legal_form_group": QBI_V3_SCF_FORM[form],
            "capital_measure": next(iter(capital_measures)),
            "proxy": next(iter(proxy_values)),
            "component_count": len(components),
            "eligible_receipts_thousands": total_receipts,
            "all_industry_receipts_thousands": all_receipts,
            # Independently rounded SOI detail rows can exceed the rounded
            # all-industry receipt total by a de minimis amount (one thousand
            # dollars in the partnership resource). Coverage is descriptive,
            # so cap that rounding artifact at one.
            "receipts_coverage": min(1.0, total_receipts / all_receipts),
            "components": components,
        }
        dispersion[form] = {
            "receipts_weighted_log_intensity_sd": weighted_log_sd,
            "receipts_weight_effective_industry_count": effective_count,
            "sigma": sigma,
        }
    return mixtures, dispersion


def build_qbi_v3_profit_margin_curves(
    employer_resource: Mapping[str, Any],
) -> tuple[list[float], dict[str, list[float]]]:
    """Extract one genuine form-level SCF empirical inverse-CDF curve per form."""

    margins = _mapping(
        employer_resource.get("profit_margin_quantiles"),
        "profit_margin_quantiles",
    )
    probabilities = [
        _probability(value, f"profit_margin_quantiles.probabilities[{index}]")
        for index, value in enumerate(
            _list(margins.get("probabilities"), "profit_margin_quantiles.probabilities")
        )
    ]
    if probabilities != sorted(probabilities) or len(set(probabilities)) != len(
        probabilities
    ):
        raise ValueError("SCF profit-margin probabilities must be strictly ordered.")
    quantile_names = [f"q{int(round(value * 100)):02d}" for value in probabilities]
    cells = [
        _mapping(cell, f"profit_margin_quantiles.cells[{index}]")
        for index, cell in enumerate(
            _list(margins.get("cells"), "profit_margin_quantiles.cells")
        )
    ]
    curves: dict[str, list[float]] = {}
    for form in QBI_V3_FORMS:
        scf_form = QBI_V3_SCF_FORM[form]
        donors = [
            cell
            for cell in cells
            if cell.get("legal_form_group") == scf_form
            and cell.get("estimate_level") == "form"
            and _mapping(cell.get("source_dimensions"), "source_dimensions").get(
                "legal_form_group"
            )
            == scf_form
            and _mapping(cell.get("source_dimensions"), "source_dimensions").get(
                "industry_code"
            )
            == "all"
        ]
        if not donors:
            raise ValueError(
                f"SCF profit-margin evidence lacks a form donor for {scf_form!r}."
            )
        donor_curves = []
        for donor in donors:
            quantiles = _mapping(donor.get("quantiles"), "quantiles")
            donor_curves.append(
                [
                    _number(quantiles[name], f"quantiles.{name}", minimum=0.0)
                    for name in quantile_names
                ]
            )
        first = donor_curves[0]
        if any(curve != first for curve in donor_curves[1:]):
            raise ValueError(f"SCF form-level margin donors disagree for {scf_form!r}.")
        if first != sorted(first) or first[0] <= 0.0:
            raise ValueError(
                f"SCF form-level margin curve is invalid for {scf_form!r}."
            )
        curves[form] = first
    return probabilities, curves


def _solve_log_odds_shift(
    base_probability: np.ndarray,
    weights: np.ndarray,
    *,
    target_zero_share: float,
) -> tuple[float, float]:
    if len(base_probability) == 0 or len(base_probability) != len(weights):
        raise ValueError("Employer calibration inputs must be nonempty and aligned.")
    if np.any(~np.isfinite(base_probability)) or np.any(
        (base_probability < 0.0) | (base_probability > 1.0)
    ):
        raise ValueError("Employer base probabilities must lie in [0, 1].")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("Employer calibration weights must be finite and positive.")
    if not 0.0 < target_zero_share < 1.0:
        raise ValueError("Employer zero-share target must lie strictly in (0, 1).")

    logits = _logit(base_probability)
    lower = _SOLVER_LOWER_BOUND
    upper = _SOLVER_UPPER_BOUND
    denominator = float(weights.sum())
    for _ in range(_SOLVER_ITERATIONS):
        midpoint = (lower + upper) / 2.0
        zero_share = float(
            np.sum(weights * (1.0 - _logistic(logits + midpoint))) / denominator
        )
        if zero_share > target_zero_share:
            lower = midpoint
        else:
            upper = midpoint
    shift = (lower + upper) / 2.0
    expected = float(np.sum(weights * (1.0 - _logistic(logits + shift))) / denominator)
    return shift, expected


def calibrate_qbi_v3_employer_shifts(
    qbi: Sequence[float] | np.ndarray,
    form_codes: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    base_probability_by_form: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Solve and persist one expected-probability log-odds shift per form."""

    qbi_values = np.asarray(qbi, dtype=np.float64)
    form_values = np.asarray(form_codes)
    weight_values = np.asarray(weights, dtype=np.float64)
    if (
        qbi_values.ndim != 1
        or form_values.ndim != 1
        or weight_values.ndim != 1
        or len({len(qbi_values), len(form_values), len(weight_values)}) != 1
    ):
        raise ValueError("QBI v3 replay QBI, forms, and weights must align.")
    if np.any(~np.isfinite(qbi_values)):
        raise ValueError("QBI v3 replay QBI must be finite.")
    if np.any(~np.isfinite(weight_values)) or np.any(weight_values < 0.0):
        raise ValueError("QBI v3 replay weights must be finite and nonnegative.")
    if weight_values.sum() <= 0.0:
        raise ValueError("QBI v3 replay weights must have positive mass.")
    valid_codes = {-1, *QBI_V3_FORM_CODES.values()}
    if not set(np.unique(form_values)).issubset(valid_codes):
        raise ValueError("QBI v3 replay contains an unknown form code.")
    positive = qbi_values > 0.0
    if not np.array_equal(positive, form_values >= 0):
        raise ValueError("Every and only positive-QBI replay row must have a form.")

    band_index = _income_band_index(qbi_values)
    form_counts: dict[str, int] = {}
    form_weights: dict[str, float] = {}
    form_band_counts: dict[str, dict[str, int]] = {}
    form_band_weights: dict[str, dict[str, float]] = {}
    base_vectors: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for form in QBI_V3_FORMS:
        code = QBI_V3_FORM_CODES[form]
        mask = form_values == code
        masks[form] = mask
        form_counts[form] = int(np.count_nonzero(mask))
        form_weights[form] = float(weight_values[mask].sum())
        if form_counts[form] == 0 or form_weights[form] <= 0.0:
            raise ValueError(f"QBI v3 replay has no positive weight for {form!r}.")
        form_probabilities = _mapping(
            base_probability_by_form.get(form),
            f"base_probability_by_form.{form}",
        )
        if set(form_probabilities) != set(SCF_INCOME_BANDS):
            raise ValueError(
                f"Employer base probabilities for {form!r} must cover all bands."
            )
        ordered_probabilities = np.asarray(
            [
                _probability(
                    form_probabilities[band],
                    f"base_probability_by_form.{form}.{band}",
                )
                for band in SCF_INCOME_BANDS
            ],
            dtype=np.float64,
        )
        base_vectors[form] = ordered_probabilities[band_index[mask]]
        form_band_counts[form] = {}
        form_band_weights[form] = {}
        for index, band in enumerate(SCF_INCOME_BANDS):
            band_mask = mask & (band_index == index)
            form_band_counts[form][band] = int(np.count_nonzero(band_mask))
            form_band_weights[form][band] = float(weight_values[band_mask].sum())

    total_form_weight = sum(form_weights.values())
    s_target = (
        _OVERALL_ZERO_EMPLOYEE_TARGET * total_form_weight
        - _SOLE_ZERO_EMPLOYEE_TARGET * form_weights["sole_proprietorship"]
        - _PARTNERSHIP_ZERO_EMPLOYEE_TARGET * form_weights["partnership"]
    ) / form_weights["s_corporation"]
    if not 0.0 < s_target < 1.0:
        raise ValueError(
            "Replay form weights imply an infeasible S-corporation zero-employee "
            f"residual {s_target}."
        )
    targets = {
        "sole_proprietorship": _SOLE_ZERO_EMPLOYEE_TARGET,
        "partnership": _PARTNERSHIP_ZERO_EMPLOYEE_TARGET,
        "s_corporation": float(s_target),
    }
    shifts: dict[str, float] = {}
    expected: dict[str, float] = {}
    for form in QBI_V3_FORMS:
        shifts[form], expected[form] = _solve_log_odds_shift(
            base_vectors[form],
            weight_values[masks[form]],
            target_zero_share=targets[form],
        )
    expected_overall = float(
        sum(form_weights[form] * expected[form] for form in QBI_V3_FORMS)
        / total_form_weight
    )
    return {
        "form_record_counts": form_counts,
        "form_record_weights": form_weights,
        "form_income_band_record_counts": form_band_counts,
        "form_income_band_weights": form_band_weights,
        "positive_qbi_record_count": int(np.count_nonzero(positive)),
        "positive_qbi_weight": float(weight_values[positive].sum()),
        "target_zero_employee_share_by_form": targets,
        "log_odds_shift_by_form": shifts,
        "expected_zero_employee_share_by_form": expected,
        "expected_overall_zero_employee_share": expected_overall,
    }


def _all_industry_wage_bills(
    wage_capital_resource: Mapping[str, Any],
) -> dict[str, float]:
    forms = _mapping(wage_capital_resource.get("forms"), "forms")
    bills: dict[str, float] = {}
    for form in QBI_V3_FORMS:
        row = _all_industry_row(_mapping(forms.get(form), f"forms.{form}"), form=form)
        raw = _mapping(
            row.get("raw_amounts_thousands"),
            f"forms.{form}.all.raw_amounts_thousands",
        )
        if form == "sole_proprietorship":
            amount = _number(
                raw.get("payroll"), f"forms.{form}.all.payroll", minimum=0.0
            )
        elif form == "partnership":
            amount = _number(
                raw.get("cost_labor"),
                f"forms.{form}.all.cost_labor",
                minimum=0.0,
            ) + _number(
                raw.get("salaries"),
                f"forms.{form}.all.salaries",
                minimum=0.0,
            )
        else:
            amount = _number(
                raw.get("officer_compensation"),
                f"forms.{form}.all.officer_compensation",
                minimum=0.0,
            ) + _number(
                raw.get("salaries"),
                f"forms.{form}.all.salaries",
                minimum=0.0,
            )
        bills[form] = float(amount * 1_000.0)
    return bills


def build_qbi_v3_assumptions_payload(
    *,
    v2_payload: Mapping[str, Any],
    employer_resource: Mapping[str, Any],
    wage_capital_resource: Mapping[str, Any],
    replay_arrays: Mapping[str, Sequence[Any]],
    person_weights: Sequence[float] | np.ndarray,
    replay_artifact: Mapping[str, Any],
    resource_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the complete deterministic QBI v3 assumptions payload."""

    validate_qbi_employer_structure_resource(employer_resource)
    validate_qbi_wage_capital_priors_resource(wage_capital_resource)

    # Import lazily so the runtime may import pure helpers from this module
    # without creating an import cycle.
    from populace.build.us_runtime.qbi_simulation import (
        QBI_SIMULATION_SOURCE_NAMES,
        QBI_SIMULATION_V2,
        QbiSimulationInputs,
        _derive_v2_qualification_flags,
        _qualified_components,
        parse_qbi_simulation_assumptions,
    )

    parent = parse_qbi_simulation_assumptions(
        v2_payload,
        qbi_simulation_version=QBI_SIMULATION_V2,
    )
    if parent.source_order != QBI_SIMULATION_SOURCE_NAMES:
        raise ValueError("QBI v3 parent source order is not recognized.")
    parent_seeds = _mapping(
        _mapping(v2_payload.get("rng"), "rng").get("seeds"),
        "rng.seeds",
    )
    for family, expected_seed in QBI_V3_RETAINED_SEEDS.items():
        if parent_seeds.get(family) != expected_seed:
            raise ValueError(f"QBI v3 must retain v2 {family} seed {expected_seed}.")

    inputs = QbiSimulationInputs.from_puf_arrays(replay_arrays)
    weights = np.asarray(person_weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) != inputs.n:
        raise ValueError("QBI v3 replay person weights must align with inputs.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError(
            "QBI v3 replay person weights must be finite, nonnegative, "
            "and have positive mass."
        )
    flags = _derive_v2_qualification_flags(inputs, assumptions=parent)
    qualified_components = _qualified_components(
        inputs,
        source_order=parent.source_order,
        qualification_flags=flags,
    )
    qbi, form_codes = assign_qbi_v3_record_forms(qualified_components)

    base_probabilities, base_levels = build_qbi_v3_employer_base_probabilities(
        employer_resource
    )
    calibration = calibrate_qbi_v3_employer_shifts(
        qbi,
        form_codes,
        weights,
        base_probability_by_form=base_probabilities,
    )
    mixtures, dispersion = build_qbi_v3_soi_mixtures(wage_capital_resource)
    margin_probabilities, margin_curves = build_qbi_v3_profit_margin_curves(
        employer_resource
    )
    wage_bills = _all_industry_wage_bills(wage_capital_resource)
    wage_lower = (
        _SOLE_PROPRIETORSHIP_ENTITY_SHARE * wage_bills["sole_proprietorship"]
        + _PARTNERSHIP_ENTITY_SHARE * wage_bills["partnership"]
        + _S_CORPORATION_ENTITY_SHARE * wage_bills["s_corporation"]
    ) / _SUPPORTED_FORM_SHARE
    wage_upper = sum(wage_bills.values())

    artifact = _exact_keys(
        replay_artifact,
        ("filename", "sha256", "bytes", "tax_unit_rows"),
        "replay_artifact",
    )
    artifact_filename = _text(artifact["filename"], "replay_artifact.filename")
    if "/" in artifact_filename or "\\" in artifact_filename:
        raise ValueError("Replay artifact filename must be a basename.")
    digests = _exact_keys(
        resource_sha256,
        ("v2_assumptions", "employer_structure", "wage_capital"),
        "resource_sha256",
    )
    replay_metadata = {
        "artifact_filename": artifact_filename,
        "artifact_sha256": _sha256(artifact["sha256"], "replay_artifact.sha256"),
        "artifact_bytes": _positive_integer(artifact["bytes"], "replay_artifact.bytes"),
        "person_rows": inputs.n,
        "tax_unit_rows": _positive_integer(
            artifact["tax_unit_rows"], "replay_artifact.tax_unit_rows"
        ),
        "person_weight_total": float(weights.sum()),
        "positive_qbi_record_count": calibration["positive_qbi_record_count"],
        "positive_qbi_weight": calibration["positive_qbi_weight"],
        "form_record_counts": calibration["form_record_counts"],
        "form_record_weights": calibration["form_record_weights"],
        "form_income_band_record_counts": calibration["form_income_band_record_counts"],
        "form_income_band_weights": calibration["form_income_band_weights"],
    }

    payload: dict[str, Any] = {
        "schema_version": QBI_V3_SCHEMA_VERSION,
        "qbi_simulation_version": QBI_V3_SIMULATION_VERSION,
        "engine": QBI_V3_ENGINE,
        "rng": {
            "bit_generator": "PCG64",
            "seeds": dict(QBI_V3_SEEDS),
        },
        "source_order": list(parent.source_order),
        "qualification_derivations": copy.deepcopy(
            v2_payload["qualification_derivations"]
        ),
        "sstb_classification": copy.deepcopy(v2_payload["sstb_classification"]),
        "evidence": {
            "v2_assumptions_resource": QBI_V3_PARENT_RESOURCE,
            "v2_assumptions_schema_version": 2,
            "v2_assumptions_sha256": _sha256(
                digests["v2_assumptions"], "resource_sha256.v2_assumptions"
            ),
            "employer_structure_resource": QBI_V3_EMPLOYER_RESOURCE,
            "employer_structure_schema_version": 1,
            "employer_structure_sha256": _sha256(
                digests["employer_structure"],
                "resource_sha256.employer_structure",
            ),
            "wage_capital_resource": QBI_V3_WAGE_CAPITAL_RESOURCE,
            "wage_capital_schema_version": 1,
            "wage_capital_sha256": _sha256(
                digests["wage_capital"], "resource_sha256.wage_capital"
            ),
        },
        "record_form": {
            "form_order": list(QBI_V3_FORMS),
            "sole_proprietorship_sources": list(parent.source_order[:-1]),
            "passthrough_source": parent.source_order[-1],
            "partnership_probability": _PARTNERSHIP_PROBABILITY,
            "s_corporation_probability": _S_CORPORATION_PROBABILITY,
            "jct_qbi_entity_shares": {
                "sole_proprietorship": _SOLE_PROPRIETORSHIP_ENTITY_SHARE,
                "partnership": _PARTNERSHIP_ENTITY_SHARE,
                "s_corporation": _S_CORPORATION_ENTITY_SHARE,
                "estate_and_trust": _ESTATE_TRUST_ENTITY_SHARE,
            },
            "rationale": (
                "A positive qualified partnership_s_corp_income component takes "
                "record-level precedence and is split using JCT QBI deduction "
                "shares 17/(17+53) and 53/(17+53). Every other positive-QBI "
                "record uses the sole-proprietorship evidence proxy. This is a "
                "latent simulation form, not an assigned observed entity type; "
                "the PUF does not identify industry or legal form within its "
                "other aggregate business-income sources."
            ),
        },
        "industry_mixture": {
            "model": "receipts_weighted_finest_classified_soi_rows",
            "eligibility": {
                "nonaggregate": True,
                "exclude_unallocable": True,
                "positive_receipts": True,
                "requires_wage_share": True,
                "requires_ubia_intensity": True,
            },
            "rationale": (
                "PUF records carry no industry. Industry heterogeneity is "
                "represented distributionally, not assigned: within each latent "
                "legal form, the record draws once from finest classified "
                "nonaggregate SOI rows with probability proportional to receipts, "
                "then uses that component's wage share and UBIA intensity. "
                "Receipts coverage is capped at one when independently rounded "
                "detail rows exceed the rounded all-industry row."
            ),
            "forms": mixtures,
        },
        "employer_presence": {
            "model": "scf_income_form_log_odds_shift",
            "income_band_order": list(SCF_INCOME_BANDS),
            "minimum_effective_n": SCF_MINIMUM_UNWEIGHTED_N,
            "industry_marginalization": (
                "Sum requested SCF counts across all eight industry bins within "
                "income band and legal form. Use the grouped income/form rate "
                "when effective n is at least 30; otherwise use the grouped "
                "form-level rate."
            ),
            "base_probability_by_form": base_probabilities,
            "base_probability_source_level_by_form": base_levels,
            "calibration": {
                "replay": replay_metadata,
                "overall_zero_employee_target": _OVERALL_ZERO_EMPLOYEE_TARGET,
                "target_zero_employee_share_by_form": calibration[
                    "target_zero_employee_share_by_form"
                ],
                "log_odds_shift_by_form": calibration["log_odds_shift_by_form"],
                "expected_zero_employee_share_by_form": calibration[
                    "expected_zero_employee_share_by_form"
                ],
                "expected_overall_zero_employee_share": calibration[
                    "expected_overall_zero_employee_share"
                ],
                "solver": {
                    "method": "bisection_on_expected_weighted_zero_share",
                    "iterations": _SOLVER_ITERATIONS,
                    "lower_bound": _SOLVER_LOWER_BOUND,
                    "upper_bound": _SOLVER_UPPER_BOUND,
                    "probability_clip": _PROBABILITY_CLIP,
                },
                "rationale": (
                    "The sole-proprietorship and partnership statements 'more "
                    "than 95%' and 'more than 80%' are encoded at the binding "
                    "0.95 and 0.80 targets. The S-corporation target is the "
                    "residual [0.842*sum(form replay weights) - 0.95*sole weight "
                    "- 0.80*partnership weight] / S-corporation weight, so the "
                    "full-artifact expected weighted zero-employee share is "
                    "exactly 0.842. A single persisted log-odds shift per form "
                    "preserves the SCF income-band shape. The simulation does "
                    "not solve calibration parameters."
                ),
            },
        },
        "profit_margin": {
            "model": "scf_form_empirical_inverse_cdf",
            "probabilities": margin_probabilities,
            "interpolation": "piecewise_linear_with_endpoint_clamp",
            "quantiles_by_form": margin_curves,
            "rationale": (
                "Receipts equal positive qualified business income divided by "
                "a seeded draw from the persisted form-level SCF empirical "
                "profit-margin inverse. Linear interpolation connects q05, q25, "
                "q50, q75, and q95; draws below q05 or above q95 clamp to the "
                "nearest endpoint. This replaces the v1/v2 invented Beta margins."
            ),
        },
        "w2": {
            "model": "employer_gate_soi_wage_share_times_receipts",
            "non_employer_wages": 0.0,
            "s_corporation_officer_compensation_included": True,
            "plausibility_band": {
                "all_industry_wage_bill_dollars_by_form": wage_bills,
                "jct_supported_form_shares": {
                    "sole_proprietorship": _SOLE_PROPRIETORSHIP_ENTITY_SHARE,
                    "partnership": _PARTNERSHIP_ENTITY_SHARE,
                    "s_corporation": _S_CORPORATION_ENTITY_SHARE,
                },
                "supported_form_share_denominator": _SUPPORTED_FORM_SHARE,
                "lower_dollars": float(wage_lower),
                "upper_dollars": float(wage_upper),
                "formula": (
                    "lower=(0.28*sole_proprietorship+0.17*partnership+"
                    "0.53*s_corporation)/0.98; "
                    "upper=sole_proprietorship+partnership+s_corporation"
                ),
                "rationale": (
                    "This provisional magnitude envelope is derived from the "
                    "persisted SOI all-industry wage bills. Its lower edge is "
                    "their supported-form JCT QBI-share-weighted composite and "
                    "its upper edge is their sum. It is a broad plausibility "
                    "diagnostic, not a confidence interval or calibration target."
                ),
            },
            "rationale": (
                "Employer records receive the latent industry's SOI wage share "
                "times SCF-margin-implied receipts; non-employers receive zero. "
                "Partnership wages include cost of labor and salaries but exclude "
                "guaranteed payments. S-corporation wages include officer "
                "compensation and salaries."
            ),
        },
        "ubia": {
            "model": "soi_industry_intensity_times_receipts_mean_one_lognormal",
            "dispersion": {
                "model": "mean_one_lognormal",
                "forms": dispersion,
                "rationale": (
                    "For each form, sigma equals the receipts-weighted standard "
                    "deviation of log SOI industry intensity divided by the "
                    "square root of the receipts-weight effective industry count "
                    "(1/sum(weight^2)). This retains modest within-component "
                    "dispersion while the latent industry mixture carries the "
                    "observed cross-industry heterogeneity. The runtime multiplier "
                    "exp(sigma*z - sigma^2/2) has mean one."
                ),
            },
            "rationale": (
                "UBIA is the latent industry's persisted intensity times "
                "SCF-margin-implied receipts and a mean-one seeded multiplier. "
                "Sole-proprietor intensity remains explicitly proxy=true because "
                "its public evidence is depreciation flow over receipts rather "
                "than an asset stock. The v1/v2 lognormal multiple of QBI is not "
                "used."
            ),
        },
        "investment": copy.deepcopy(v2_payload["investment"]),
    }
    validate_qbi_v3_assumptions_payload(
        payload,
        expected_v2_payload=v2_payload,
    )
    return payload


def _load_packaged_v2_payload() -> Mapping[str, Any]:
    resource = files("populace.build.us").joinpath(QBI_V3_PARENT_RESOURCE)
    return _mapping(json.loads(resource.read_text(encoding="utf-8")), "packaged v2")


def validate_qbi_v3_assumptions_payload(
    payload: Mapping[str, Any],
    *,
    expected_v2_payload: Mapping[str, Any] | None = None,
) -> None:
    """Strictly validate the committed QBI v3 assumptions contract."""

    root = _exact_keys(
        payload,
        (
            "schema_version",
            "qbi_simulation_version",
            "engine",
            "rng",
            "source_order",
            "qualification_derivations",
            "sstb_classification",
            "evidence",
            "record_form",
            "industry_mixture",
            "employer_presence",
            "profit_margin",
            "w2",
            "ubia",
            "investment",
        ),
        "QBI v3 assumptions",
    )
    if root["schema_version"] != QBI_V3_SCHEMA_VERSION:
        raise ValueError("QBI v3 schema_version must equal 3.")
    if root["qbi_simulation_version"] != QBI_V3_SIMULATION_VERSION:
        raise ValueError("QBI v3 qbi_simulation_version must equal 3.")
    if root["engine"] != QBI_V3_ENGINE:
        raise ValueError("QBI v3 engine is not recognized.")

    parent = expected_v2_payload or _load_packaged_v2_payload()
    for key in (
        "source_order",
        "qualification_derivations",
        "sstb_classification",
        "investment",
    ):
        if root[key] != parent[key]:
            raise ValueError(f"QBI v3 must copy v2 {key} unchanged.")

    rng = _exact_keys(root["rng"], ("bit_generator", "seeds"), "rng")
    if rng["bit_generator"] != "PCG64":
        raise ValueError("QBI v3 requires PCG64.")
    seeds = _exact_keys(rng["seeds"], set(QBI_V3_SEEDS), "rng.seeds")
    parsed_seeds = {
        family: _nonnegative_integer(seeds[family], f"rng.seeds.{family}")
        for family in QBI_V3_SEEDS
    }
    if parsed_seeds != QBI_V3_SEEDS:
        raise ValueError("QBI v3 seed assignments do not match the contract.")
    if len(set(parsed_seeds.values())) != len(parsed_seeds):
        raise ValueError("QBI v3 family seeds must be distinct.")

    evidence = _exact_keys(
        root["evidence"],
        (
            "v2_assumptions_resource",
            "v2_assumptions_schema_version",
            "v2_assumptions_sha256",
            "employer_structure_resource",
            "employer_structure_schema_version",
            "employer_structure_sha256",
            "wage_capital_resource",
            "wage_capital_schema_version",
            "wage_capital_sha256",
        ),
        "evidence",
    )
    expected_evidence = {
        "v2_assumptions_resource": QBI_V3_PARENT_RESOURCE,
        "v2_assumptions_schema_version": 2,
        "employer_structure_resource": QBI_V3_EMPLOYER_RESOURCE,
        "employer_structure_schema_version": 1,
        "wage_capital_resource": QBI_V3_WAGE_CAPITAL_RESOURCE,
        "wage_capital_schema_version": 1,
    }
    for key, expected in expected_evidence.items():
        if evidence[key] != expected:
            raise ValueError(f"QBI v3 evidence {key} is not recognized.")
    for key in (
        "v2_assumptions_sha256",
        "employer_structure_sha256",
        "wage_capital_sha256",
    ):
        _sha256(evidence[key], f"evidence.{key}")

    record_form = _exact_keys(
        root["record_form"],
        (
            "form_order",
            "sole_proprietorship_sources",
            "passthrough_source",
            "partnership_probability",
            "s_corporation_probability",
            "jct_qbi_entity_shares",
            "rationale",
        ),
        "record_form",
    )
    if record_form["form_order"] != list(QBI_V3_FORMS):
        raise ValueError("QBI v3 form_order is not recognized.")
    source_order = list(root["source_order"])
    if record_form["sole_proprietorship_sources"] != source_order[:-1]:
        raise ValueError("QBI v3 sole-proprietorship sources are not recognized.")
    if record_form["passthrough_source"] != source_order[-1]:
        raise ValueError("QBI v3 passthrough source is not recognized.")
    if not math.isclose(
        _probability(
            record_form["partnership_probability"],
            "record_form.partnership_probability",
        ),
        _PARTNERSHIP_PROBABILITY,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("QBI v3 partnership probability must equal 17/70.")
    if not math.isclose(
        _probability(
            record_form["s_corporation_probability"],
            "record_form.s_corporation_probability",
        ),
        _S_CORPORATION_PROBABILITY,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("QBI v3 S-corporation probability must equal 53/70.")
    shares = _exact_keys(
        record_form["jct_qbi_entity_shares"],
        ("sole_proprietorship", "partnership", "s_corporation", "estate_and_trust"),
        "record_form.jct_qbi_entity_shares",
    )
    expected_shares = {
        "sole_proprietorship": _SOLE_PROPRIETORSHIP_ENTITY_SHARE,
        "partnership": _PARTNERSHIP_ENTITY_SHARE,
        "s_corporation": _S_CORPORATION_ENTITY_SHARE,
        "estate_and_trust": _ESTATE_TRUST_ENTITY_SHARE,
    }
    if {
        key: _probability(value, f"entity_shares.{key}")
        for key, value in shares.items()
    } != expected_shares:
        raise ValueError("QBI v3 JCT entity shares are not recognized.")
    _text(record_form["rationale"], "record_form.rationale")

    mixture = _exact_keys(
        root["industry_mixture"],
        ("model", "eligibility", "rationale", "forms"),
        "industry_mixture",
    )
    if mixture["model"] != "receipts_weighted_finest_classified_soi_rows":
        raise ValueError("QBI v3 industry mixture model is not recognized.")
    eligibility = _exact_keys(
        mixture["eligibility"],
        (
            "nonaggregate",
            "exclude_unallocable",
            "positive_receipts",
            "requires_wage_share",
            "requires_ubia_intensity",
        ),
        "industry_mixture.eligibility",
    )
    if any(value is not True for value in eligibility.values()):
        raise ValueError("QBI v3 industry eligibility flags must all be true.")
    _text(mixture["rationale"], "industry_mixture.rationale")
    mixture_forms = _exact_keys(
        mixture["forms"], set(QBI_V3_FORMS), "industry_mixture.forms"
    )
    for form in QBI_V3_FORMS:
        form_payload = _exact_keys(
            mixture_forms[form],
            (
                "tax_year",
                "scf_legal_form_group",
                "capital_measure",
                "proxy",
                "component_count",
                "eligible_receipts_thousands",
                "all_industry_receipts_thousands",
                "receipts_coverage",
                "components",
            ),
            f"industry_mixture.forms.{form}",
        )
        if form_payload["scf_legal_form_group"] != QBI_V3_SCF_FORM[form]:
            raise ValueError(f"QBI v3 {form} has the wrong SCF legal form.")
        if not isinstance(form_payload["tax_year"], int):
            raise ValueError(f"QBI v3 {form} tax_year must be an integer.")
        _text(form_payload["capital_measure"], f"{form}.capital_measure")
        if not isinstance(form_payload["proxy"], bool):
            raise ValueError(f"QBI v3 {form} proxy must be boolean.")
        components = _list(form_payload["components"], f"{form}.components")
        if _positive_integer(
            form_payload["component_count"], f"{form}.component_count"
        ) != len(components):
            raise ValueError(f"QBI v3 {form} component count is inconsistent.")
        probabilities: list[float] = []
        seen: set[str] = set()
        for index, raw_component in enumerate(components):
            component = _exact_keys(
                raw_component,
                (
                    "industry_key",
                    "published_label",
                    "probability",
                    "wage_share",
                    "ubia_intensity",
                    "proxy",
                    "capital_measure",
                ),
                f"{form}.components[{index}]",
            )
            key = _text(component["industry_key"], f"{form}.industry_key")
            if key in seen:
                raise ValueError(f"QBI v3 {form} duplicates industry {key!r}.")
            seen.add(key)
            _text(component["published_label"], f"{form}.published_label")
            probabilities.append(
                _probability(component["probability"], f"{form}.probability")
            )
            _number(component["wage_share"], f"{form}.wage_share", minimum=0.0)
            if (
                _number(
                    component["ubia_intensity"],
                    f"{form}.ubia_intensity",
                    minimum=0.0,
                )
                <= 0.0
            ):
                raise ValueError(f"QBI v3 {form} UBIA intensity must be positive.")
            if component["proxy"] is not form_payload["proxy"]:
                raise ValueError(f"QBI v3 {form} component proxy is inconsistent.")
            if component["capital_measure"] != form_payload["capital_measure"]:
                raise ValueError(
                    f"QBI v3 {form} component capital measure is inconsistent."
                )
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"QBI v3 {form} mixture probabilities must sum to one.")
        eligible_receipts = _number(
            form_payload["eligible_receipts_thousands"],
            f"{form}.eligible_receipts_thousands",
            minimum=0.0,
        )
        all_receipts = _number(
            form_payload["all_industry_receipts_thousands"],
            f"{form}.all_industry_receipts_thousands",
            minimum=0.0,
        )
        coverage = _number(
            form_payload["receipts_coverage"], f"{form}.receipts_coverage", minimum=0.0
        )
        if all_receipts <= 0.0:
            raise ValueError(f"QBI v3 {form} all-industry receipts must be positive.")
        expected_coverage = min(1.0, eligible_receipts / all_receipts)
        if not math.isclose(
            coverage,
            expected_coverage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"QBI v3 {form} receipts coverage is inconsistent.")

    employer = _exact_keys(
        root["employer_presence"],
        (
            "model",
            "income_band_order",
            "minimum_effective_n",
            "industry_marginalization",
            "base_probability_by_form",
            "base_probability_source_level_by_form",
            "calibration",
        ),
        "employer_presence",
    )
    if employer["model"] != "scf_income_form_log_odds_shift":
        raise ValueError("QBI v3 employer model is not recognized.")
    if employer["income_band_order"] != list(SCF_INCOME_BANDS):
        raise ValueError("QBI v3 employer income bands are not recognized.")
    if employer["minimum_effective_n"] != SCF_MINIMUM_UNWEIGHTED_N:
        raise ValueError("QBI v3 employer minimum n must equal 30.")
    _text(employer["industry_marginalization"], "industry_marginalization")
    base = _exact_keys(
        employer["base_probability_by_form"],
        set(QBI_V3_FORMS),
        "base_probability_by_form",
    )
    levels = _exact_keys(
        employer["base_probability_source_level_by_form"],
        set(QBI_V3_FORMS),
        "base_probability_source_level_by_form",
    )
    for form in QBI_V3_FORMS:
        form_base = _exact_keys(
            base[form], set(SCF_INCOME_BANDS), f"base_probability_by_form.{form}"
        )
        form_levels = _exact_keys(
            levels[form],
            set(SCF_INCOME_BANDS),
            f"base_probability_source_level_by_form.{form}",
        )
        for band in SCF_INCOME_BANDS:
            _probability(form_base[band], f"base.{form}.{band}")
            if form_levels[band] not in {"income_form", "form"}:
                raise ValueError("QBI v3 employer source level is not recognized.")

    calibration = _exact_keys(
        employer["calibration"],
        (
            "replay",
            "overall_zero_employee_target",
            "target_zero_employee_share_by_form",
            "log_odds_shift_by_form",
            "expected_zero_employee_share_by_form",
            "expected_overall_zero_employee_share",
            "solver",
            "rationale",
        ),
        "employer_presence.calibration",
    )
    if calibration["overall_zero_employee_target"] != _OVERALL_ZERO_EMPLOYEE_TARGET:
        raise ValueError("QBI v3 overall zero-employee target must equal 0.842.")
    targets = _exact_keys(
        calibration["target_zero_employee_share_by_form"],
        set(QBI_V3_FORMS),
        "target_zero_employee_share_by_form",
    )
    shifts = _exact_keys(
        calibration["log_odds_shift_by_form"],
        set(QBI_V3_FORMS),
        "log_odds_shift_by_form",
    )
    achieved = _exact_keys(
        calibration["expected_zero_employee_share_by_form"],
        set(QBI_V3_FORMS),
        "expected_zero_employee_share_by_form",
    )
    parsed_targets = {
        form: _probability(targets[form], f"targets.{form}") for form in QBI_V3_FORMS
    }
    if (
        parsed_targets["sole_proprietorship"] != _SOLE_ZERO_EMPLOYEE_TARGET
        or parsed_targets["partnership"] != _PARTNERSHIP_ZERO_EMPLOYEE_TARGET
    ):
        raise ValueError("QBI v3 sole/partnership zero targets are not recognized.")
    parsed_shifts = {
        form: _number(shifts[form], f"shifts.{form}") for form in QBI_V3_FORMS
    }
    parsed_achieved = {
        form: _probability(achieved[form], f"achieved.{form}") for form in QBI_V3_FORMS
    }
    for form in QBI_V3_FORMS:
        if not math.isclose(
            parsed_achieved[form],
            parsed_targets[form],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"QBI v3 {form} expected calibration misses its target.")
    replay = _exact_keys(
        calibration["replay"],
        (
            "artifact_filename",
            "artifact_sha256",
            "artifact_bytes",
            "person_rows",
            "tax_unit_rows",
            "person_weight_total",
            "positive_qbi_record_count",
            "positive_qbi_weight",
            "form_record_counts",
            "form_record_weights",
            "form_income_band_record_counts",
            "form_income_band_weights",
        ),
        "employer_presence.calibration.replay",
    )
    filename = _text(replay["artifact_filename"], "replay.artifact_filename")
    if "/" in filename or "\\" in filename:
        raise ValueError("QBI v3 replay artifact filename must be a basename.")
    _sha256(replay["artifact_sha256"], "replay.artifact_sha256")
    _positive_integer(replay["artifact_bytes"], "replay.artifact_bytes")
    _positive_integer(replay["person_rows"], "replay.person_rows")
    _positive_integer(replay["tax_unit_rows"], "replay.tax_unit_rows")
    _number(replay["person_weight_total"], "replay.person_weight_total", minimum=0.0)
    positive_count = _positive_integer(
        replay["positive_qbi_record_count"], "replay.positive_qbi_record_count"
    )
    positive_weight = _number(
        replay["positive_qbi_weight"], "replay.positive_qbi_weight", minimum=0.0
    )
    form_counts = _exact_keys(
        replay["form_record_counts"], set(QBI_V3_FORMS), "replay.form_record_counts"
    )
    form_weights = _exact_keys(
        replay["form_record_weights"], set(QBI_V3_FORMS), "replay.form_record_weights"
    )
    band_counts = _exact_keys(
        replay["form_income_band_record_counts"],
        set(QBI_V3_FORMS),
        "replay.form_income_band_record_counts",
    )
    band_weights = _exact_keys(
        replay["form_income_band_weights"],
        set(QBI_V3_FORMS),
        "replay.form_income_band_weights",
    )
    parsed_form_counts = {
        form: _positive_integer(form_counts[form], f"form_counts.{form}")
        for form in QBI_V3_FORMS
    }
    parsed_form_weights = {
        form: _number(form_weights[form], f"form_weights.{form}", minimum=0.0)
        for form in QBI_V3_FORMS
    }
    if positive_count != sum(parsed_form_counts.values()) or not math.isclose(
        positive_weight,
        sum(parsed_form_weights.values()),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("QBI v3 replay positive totals do not match form totals.")

    recomputed_expected: dict[str, float] = {}
    for form in QBI_V3_FORMS:
        counts = _exact_keys(
            band_counts[form], set(SCF_INCOME_BANDS), f"band_counts.{form}"
        )
        weights = _exact_keys(
            band_weights[form], set(SCF_INCOME_BANDS), f"band_weights.{form}"
        )
        parsed_counts = {
            band: _nonnegative_integer(counts[band], f"band_counts.{form}.{band}")
            for band in SCF_INCOME_BANDS
        }
        parsed_weights = {
            band: _number(weights[band], f"band_weights.{form}.{band}", minimum=0.0)
            for band in SCF_INCOME_BANDS
        }
        if sum(parsed_counts.values()) != parsed_form_counts[form] or not math.isclose(
            sum(parsed_weights.values()),
            parsed_form_weights[form],
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"QBI v3 replay {form} band totals are inconsistent.")
        numerator = 0.0
        for band in SCF_INCOME_BANDS:
            adjusted_employer = float(
                _logistic(
                    _logit(_probability(base[form][band], f"base.{form}.{band}"))
                    + parsed_shifts[form]
                )
            )
            numerator += parsed_weights[band] * (1.0 - adjusted_employer)
        recomputed_expected[form] = numerator / parsed_form_weights[form]
        if not math.isclose(
            recomputed_expected[form],
            parsed_achieved[form],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"QBI v3 replay {form} weights do not reproduce its calibration."
            )
    expected_s_target = (
        _OVERALL_ZERO_EMPLOYEE_TARGET * sum(parsed_form_weights.values())
        - _SOLE_ZERO_EMPLOYEE_TARGET * parsed_form_weights["sole_proprietorship"]
        - _PARTNERSHIP_ZERO_EMPLOYEE_TARGET * parsed_form_weights["partnership"]
    ) / parsed_form_weights["s_corporation"]
    if not math.isclose(
        parsed_targets["s_corporation"],
        expected_s_target,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("QBI v3 S-corporation residual target is inconsistent.")
    expected_overall = sum(
        parsed_form_weights[form] * parsed_achieved[form] for form in QBI_V3_FORMS
    ) / sum(parsed_form_weights.values())
    if not math.isclose(
        _probability(
            calibration["expected_overall_zero_employee_share"],
            "expected_overall_zero_employee_share",
        ),
        expected_overall,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        expected_overall,
        _OVERALL_ZERO_EMPLOYEE_TARGET,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("QBI v3 expected overall zero share is inconsistent.")
    solver = _exact_keys(
        calibration["solver"],
        (
            "method",
            "iterations",
            "lower_bound",
            "upper_bound",
            "probability_clip",
        ),
        "employer_presence.calibration.solver",
    )
    if (
        solver["method"] != "bisection_on_expected_weighted_zero_share"
        or solver["iterations"] != _SOLVER_ITERATIONS
        or solver["lower_bound"] != _SOLVER_LOWER_BOUND
        or solver["upper_bound"] != _SOLVER_UPPER_BOUND
        or solver["probability_clip"] != _PROBABILITY_CLIP
    ):
        raise ValueError("QBI v3 employer solver metadata is not recognized.")
    _text(calibration["rationale"], "employer calibration rationale")

    margin = _exact_keys(
        root["profit_margin"],
        ("model", "probabilities", "interpolation", "quantiles_by_form", "rationale"),
        "profit_margin",
    )
    if (
        margin["model"] != "scf_form_empirical_inverse_cdf"
        or margin["interpolation"] != "piecewise_linear_with_endpoint_clamp"
    ):
        raise ValueError("QBI v3 profit-margin model is not recognized.")
    margin_probabilities = [
        _probability(value, f"profit_margin.probabilities[{index}]")
        for index, value in enumerate(_list(margin["probabilities"], "margin probs"))
    ]
    if margin_probabilities != sorted(margin_probabilities) or len(
        set(margin_probabilities)
    ) != len(margin_probabilities):
        raise ValueError("QBI v3 margin probabilities must be strictly ordered.")
    margin_curves = _exact_keys(
        margin["quantiles_by_form"], set(QBI_V3_FORMS), "quantiles_by_form"
    )
    for form in QBI_V3_FORMS:
        curve = [
            _number(value, f"margin.{form}[{index}]", minimum=0.0)
            for index, value in enumerate(_list(margin_curves[form], f"margin.{form}"))
        ]
        if len(curve) != len(margin_probabilities) or curve != sorted(curve):
            raise ValueError(f"QBI v3 {form} margin curve is invalid.")
        if curve[0] <= 0.0:
            raise ValueError(f"QBI v3 {form} margin curve must be positive.")
    _text(margin["rationale"], "profit_margin.rationale")

    w2 = _exact_keys(
        root["w2"],
        (
            "model",
            "non_employer_wages",
            "s_corporation_officer_compensation_included",
            "plausibility_band",
            "rationale",
        ),
        "w2",
    )
    if (
        w2["model"] != "employer_gate_soi_wage_share_times_receipts"
        or w2["non_employer_wages"] != 0.0
        or w2["s_corporation_officer_compensation_included"] is not True
    ):
        raise ValueError("QBI v3 W-2 model is not recognized.")
    wage_band = _exact_keys(
        w2["plausibility_band"],
        (
            "all_industry_wage_bill_dollars_by_form",
            "jct_supported_form_shares",
            "supported_form_share_denominator",
            "lower_dollars",
            "upper_dollars",
            "formula",
            "rationale",
        ),
        "w2.plausibility_band",
    )
    bills = _exact_keys(
        wage_band["all_industry_wage_bill_dollars_by_form"],
        set(QBI_V3_FORMS),
        "wage bills",
    )
    parsed_bills = {
        form: _number(bills[form], f"wage_bills.{form}", minimum=0.0)
        for form in QBI_V3_FORMS
    }
    band_shares = _exact_keys(
        wage_band["jct_supported_form_shares"],
        set(QBI_V3_FORMS),
        "wage band shares",
    )
    parsed_band_shares = {
        form: _probability(band_shares[form], f"wage_band_shares.{form}")
        for form in QBI_V3_FORMS
    }
    expected_band_shares = {
        "sole_proprietorship": _SOLE_PROPRIETORSHIP_ENTITY_SHARE,
        "partnership": _PARTNERSHIP_ENTITY_SHARE,
        "s_corporation": _S_CORPORATION_ENTITY_SHARE,
    }
    if parsed_band_shares != expected_band_shares:
        raise ValueError("QBI v3 wage-band JCT shares are not recognized.")
    denominator = _probability(
        wage_band["supported_form_share_denominator"],
        "supported_form_share_denominator",
    )
    if denominator != _SUPPORTED_FORM_SHARE:
        raise ValueError("QBI v3 supported-form denominator must equal 0.98.")
    expected_lower = (
        sum(parsed_band_shares[form] * parsed_bills[form] for form in QBI_V3_FORMS)
        / denominator
    )
    expected_upper = sum(parsed_bills.values())
    if not math.isclose(
        _number(wage_band["lower_dollars"], "wage lower", minimum=0.0),
        expected_lower,
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        _number(wage_band["upper_dollars"], "wage upper", minimum=0.0),
        expected_upper,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("QBI v3 W-2 plausibility band arithmetic is inconsistent.")
    _text(wage_band["formula"], "w2.plausibility_band.formula")
    _text(wage_band["rationale"], "w2.plausibility_band.rationale")
    _text(w2["rationale"], "w2.rationale")

    ubia = _exact_keys(root["ubia"], ("model", "dispersion", "rationale"), "ubia")
    if ubia["model"] != "soi_industry_intensity_times_receipts_mean_one_lognormal":
        raise ValueError("QBI v3 UBIA model is not recognized.")
    ubia_dispersion = _exact_keys(
        ubia["dispersion"], ("model", "forms", "rationale"), "ubia.dispersion"
    )
    if ubia_dispersion["model"] != "mean_one_lognormal":
        raise ValueError("QBI v3 UBIA dispersion model is not recognized.")
    dispersion_forms = _exact_keys(
        ubia_dispersion["forms"], set(QBI_V3_FORMS), "ubia.dispersion.forms"
    )
    for form in QBI_V3_FORMS:
        values = _exact_keys(
            dispersion_forms[form],
            (
                "receipts_weighted_log_intensity_sd",
                "receipts_weight_effective_industry_count",
                "sigma",
            ),
            f"ubia.dispersion.forms.{form}",
        )
        observed_sd = _number(
            values["receipts_weighted_log_intensity_sd"],
            f"ubia.{form}.observed_sd",
            minimum=0.0,
        )
        effective_count = _number(
            values["receipts_weight_effective_industry_count"],
            f"ubia.{form}.effective_count",
            minimum=0.0,
        )
        sigma = _number(values["sigma"], f"ubia.{form}.sigma", minimum=0.0)
        if effective_count <= 0.0 or not math.isclose(
            sigma,
            observed_sd / math.sqrt(effective_count),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"QBI v3 {form} UBIA dispersion is inconsistent.")
    _text(ubia_dispersion["rationale"], "ubia.dispersion.rationale")
    _text(ubia["rationale"], "ubia.rationale")
