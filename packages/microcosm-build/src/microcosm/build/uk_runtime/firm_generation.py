"""Experimental UK firm-population generator migrated from firm-microsim-paper.

This module ports the paper repository's ONS/HMRC firm generator into
``microcosm.build.uk_runtime`` without carrying the paper repo's processed CSVs.
Callers should pass Ledger consumer facts through
``uk_firm_source_data_from_ledger_facts``. The explicit source-table reader is
kept only for migration comparisons against the paper repository.

Scope is intentionally narrow and labelled experimental. Microcosm production
calibration targets must be materialized from Ledger target profiles; the
processed ONS/HMRC tables accepted here are a migration harness only, not a
production target source. VAT liability must be supplied by configured rules
(:class:`~microcosm.build.uk_runtime.firm_rules.UKFirmRules`, or a legacy
:class:`UKFirmVATRuleEvaluator` such as the Axiom RuleSpec evaluator), so this
module does not claim production firm microsimulation support without them.

The pipeline follows the paper generator (``firm_microsim.generate`` at paper
commit aa1ce07):

1. draw ONS-frame firms by sector and turnover band, then input costs and,
   when a sampling fraction is below one, stratified thinning whose base
   weights become the solve's initial weights;
2. append the HMRC negative/zero-turnover traders *before* calibration, as an
   out-of-frame stratum (``in_frame`` False, base weight one);
3. draw sector-conditional employment for every row;
4. evaluate VAT liability-if-registered through the rules seam;
5. calibrate on two universes (paper issue #37): ONS-frame rows (population,
   employment) count frame rows only; HMRC registered-trader rows (counts by
   band and sector, net VAT by band) weight every row by its band's
   registration propensity;
6. assign VAT scope and registration from the calibrated weights.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import Tensor

from microcosm.build.firms import draw_in_band
from microcosm.build.uk_runtime.firm_bpe_turnover import (
    OPEN_BAND_LOWER_K,
    draw_uk_firm_open_band_turnover,
    fit_uk_firm_employment_turnover_rho,
    repair_uk_firm_employment_turnover,
    uk_firm_open_band_tail_parameters,
    uk_firm_open_band_uniforms,
)
from microcosm.build.uk_runtime.firm_rules import (
    LegacyVATEvaluatorRules,
    UKFirmRules,
    align_rule_outputs,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import build_candidate_rules_table
from microcosm.build.uk_runtime.firm_universe import (
    HMRC_BAND_NAMES,
    assign_vat_scope_and_registration,
    registration_propensity,
    stratified_thin,
)
from microcosm.calibrate.solve import (
    CONSERVE_MASS,
    FREE_MASS,
    default_target_loss_scales,
    relative_error_loss,
    solve_constraint_system,
)

logger = logging.getLogger(__name__)

VINTAGES: dict[str, dict[str, float]] = {
    "2023-24": {"threshold": 85.0},
    "2024-25": {"threshold": 90.0},
}
DEFAULT_VINTAGE = "2023-24"

INPUT_FILES: dict[str, str] = {
    "ons_turnover": "ons_firm_turnover.csv",
    "ons_employment": "ons_firm_employment.csv",
    "hmrc_population_band": "hmrc_vat_population_by_turnover_band.csv",
    "hmrc_population_sector": "hmrc_vat_population_by_sector.csv",
    "hmrc_liability_band": "hmrc_vat_liability_by_turnover_band.csv",
    "hmrc_liability_sector": "hmrc_vat_liability_by_sector.csv",
}
UK_FIRM_TARGET_IDS: dict[str, str] = {
    "ons_turnover": "ons.uk_business.enterprise_count.turnover_bands",
    "ons_employment": "ons.uk_business.enterprise_count.employment_bands",
    "hmrc_population_band": "hmrc.vat.registered_trader_count.turnover_bands",
    "hmrc_liability_band": "hmrc.vat.net_liability.turnover_bands",
    "ons_sic_turnover": "ons.uk_business.enterprise_count.sic_turnover_bands",
    "ons_sic_employment": "ons.uk_business.enterprise_count.sic_employment_bands",
    "hmrc_population_sic": "hmrc.vat.registered_trader_count.sic_sectors",
    "hmrc_liability_sic": "hmrc.vat.net_liability.sic_sectors",
}


def load_uk_firms_contract() -> dict[str, Any]:
    """Load the packaged UK firm Ledger target contract."""

    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_firms_targets.json")
        .read_text()
    )
    return json.loads(payload)


@dataclass(frozen=True)
class UKFirmLedgerTargetProfile:
    """Ledger record-set selectors for the ``uk_firms`` target profile."""

    ons_turnover_record_set: str
    ons_employment_record_set: str
    hmrc_population_band_record_set: str
    hmrc_liability_band_record_set: str
    ons_sic_turnover_record_set: str
    ons_sic_employment_record_set: str
    hmrc_population_sic_record_set: str
    hmrc_liability_sic_record_set: str


DEFAULT_UK_FIRM_TARGET_PROFILE = UKFirmLedgerTargetProfile(
    ons_turnover_record_set="ons.uk_business.cy2025.enterprise_count.by_turnover_band",
    ons_employment_record_set="ons.uk_business.cy2025.enterprise_count.by_employment_band",
    hmrc_population_band_record_set=(
        "hmrc.vat.fy2024_25.registered_trader_count.by_turnover_band"
    ),
    hmrc_liability_band_record_set=(
        "hmrc.vat.fy2024_25.net_liability.by_turnover_band"
    ),
    ons_sic_turnover_record_set=(
        "ons.uk_business.cy2025.enterprise_count.by_sic_turnover_band"
    ),
    ons_sic_employment_record_set=(
        "ons.uk_business.cy2025.enterprise_count.by_sic_employment_band"
    ),
    hmrc_population_sic_record_set=(
        "hmrc.vat.fy2024_25.registered_trader_count.by_sic"
    ),
    hmrc_liability_sic_record_set="hmrc.vat.fy2024_25.net_liability.by_sic",
)

_UK_FIRM_TARGET_PROFILE_FIELDS: dict[str, str] = {
    "ons_turnover": "ons_turnover_record_set",
    "ons_employment": "ons_employment_record_set",
    "hmrc_population_band": "hmrc_population_band_record_set",
    "hmrc_liability_band": "hmrc_liability_band_record_set",
    "ons_sic_turnover": "ons_sic_turnover_record_set",
    "ons_sic_employment": "ons_sic_employment_record_set",
    "hmrc_population_sic": "hmrc_population_sic_record_set",
    "hmrc_liability_sic": "hmrc_liability_sic_record_set",
}

# These maps translate Ledger value IDs into the generator's temporary support
# matrix. The source selectors above are profile-owned; support layout should be
# the next piece lifted into a declarative spec.
LEDGER_ONS_TURNOVER_BANDS: dict[str, str] = {
    "0_49k": "0-49",
    "50_99k": "50-99",
    "100_249k": "100-249",
    "250_499k": "250-499",
    "500_999k": "500-999",
    "1000_4999k": "1000-4999",
    "5000k_plus": "5000+",
}
LEDGER_ONS_EMPLOYMENT_BANDS: dict[str, str] = {
    "0_4": "0-4",
    "5_9": "5-9",
    "10_19": "10-19",
    "20_49": "20-49",
    "50_99": "50-99",
    "100_249": "100-249",
    "250_plus": "250+",
}
LEDGER_HMRC_BANDS: dict[str, str] = {
    "negative_or_zero": "Negative_or_Zero",
    "1_to_threshold": "£1_to_Threshold",
    "threshold_to_150k": "£Threshold_to_£150k",
    "150k_to_300k": "£150k_to_£300k",
    "300k_to_500k": "£300k_to_£500k",
    "500k_to_1m": "£500k_to_£1m",
    "1m_to_10m": "£1m_to_£10m",
    "greater_than_10m": "Greater_than_£10m",
}

# Chronicle-era spellings of the three band maps above. These identifiers only
# look like environment variables to a grep — they are plain support-layout
# data, so they take aliases rather than the dual-read window in
# :mod:`microcosm.build.logbook_env`. The ledger-era names stay: they are what
# the generator below imports, and the value ids they translate are published
# Chronicle ids, unaffected by the rename.
CHRONICLE_ONS_TURNOVER_BANDS = LEDGER_ONS_TURNOVER_BANDS
CHRONICLE_ONS_EMPLOYMENT_BANDS = LEDGER_ONS_EMPLOYMENT_BANDS
CHRONICLE_HMRC_BANDS = LEDGER_HMRC_BANDS

#: HMRC VAT turnover bands, in ``band_idx`` order; defined once in
#: :mod:`~microcosm.build.uk_runtime.firm_universe`.
HMRC_BAND_COLUMNS: tuple[str, ...] = HMRC_BAND_NAMES
VAT_LIABILITY_BANDS: tuple[str, ...] = HMRC_BAND_COLUMNS[1:]
#: Net-VAT bands that enter the calibration. ``£1_to_Threshold`` is reported
#: but not calibrated, as in the paper: its HMRC total is remitted by voluntary
#: registrants whose input-reclaim-driven net remittances the
#: liability-if-registered model does not represent, and imposing it crushes
#: weights just below the threshold.
VAT_LIABILITY_BANDS_CALIBRATED: tuple[str, ...] = VAT_LIABILITY_BANDS[1:]
EMPLOYMENT_BANDS: tuple[str, ...] = (
    "0-4",
    "5-9",
    "10-19",
    "20-49",
    "50-99",
    "100-249",
    "250+",
)

#: ONS turnover bands as ``(inclusive lower, exclusive upper, midpoint)`` in
#: thousands of pounds. The published integer labels are contiguous
#: continuous intervals: "50-99" is [50, 100), as in the paper generator.
ONS_TURNOVER_BANDS: dict[str, tuple[float, float, float]] = {
    "0-49": (0.0, 50.0, 25.0),
    "50-99": (50.0, 100.0, 75.0),
    "100-249": (100.0, 250.0, 175.0),
    "250-499": (250.0, 500.0, 375.0),
    "500-999": (500.0, 1000.0, 750.0),
    "1000-4999": (1000.0, 5000.0, 3000.0),
    "5000+": (5000.0, 50000.0, 15000.0),
}
#: The open "5000+" band is drawn log-uniform on [5m, 50m) (paper issue #40):
#: a uniform draw put more enterprises above 10m than HMRC counts traders.
ONS_OPEN_TURNOVER_BAND = "5000+"
#: Floor for drawn turnover in thousands of pounds, so frame firms always have
#: positive turnover (the appended HMRC traders alone carry zero).
MIN_DRAWN_TURNOVER_K = 0.1
ONS_EMPLOYMENT_BANDS: dict[str, tuple[float, float, float]] = {
    "0-4": (1.0, 4.0, 2.5),
    "5-9": (5.0, 9.0, 7.0),
    "10-19": (10.0, 19.0, 14.5),
    "20-49": (20.0, 49.0, 34.5),
    "50-99": (50.0, 99.0, 74.5),
    "100-249": (100.0, 249.0, 174.5),
    "250+": (250.0, 2000.0, 400.0),
}

#: The legacy draw for the open 250+ band is the paper's lognormal (median
#: 400, log-sd 0.8) clamped to the band's ``[250, 2000]`` above, so no firm
#: exceeds 2,000 staff. Setting ``UKFirmGenerationConfig.large_employment_mean``
#: (for example to :data:`LARGE_EMPLOYMENT_MEAN`) replaces the clamp with a
#: truncated Pareto on ``[250, LARGE_EMPLOYMENT_CAP]`` with that mean.
#:
#: Mean headcount of a private-sector business with 250 or more employees:
#: 11,115 thousand employment over 8,250 businesses, DBT "Business
#: population estimates 2024" detailed tables, Table 1 (UK private sector,
#: start 2024; rows 250-499, 500-999 and 1000 or more). The figures and their
#: provenance are fixtured in ``tests/fixtures/uk/firm_2024_25_size/``.
LARGE_EMPLOYMENT_MEAN = 11_115_000.0 / 8_250.0
#: Upper limit of the Pareto 250+ headcount draw. It sits above the largest
#: UK private employers' headcounts, so it bounds the draw's variance without
#: acting as the legacy clamp at 2,000.
LARGE_EMPLOYMENT_CAP = 500_000.0


#: Shape bracket of the Pareto bisection. The truncated mean falls as the
#: shape rises, so the bracket's ends bound the means a shape can reach.
LARGE_EMPLOYMENT_SHAPE_BRACKET = (0.05, 20.0)


def _large_employment_truncated_mean(shape: float, lower: float, cap: float) -> float:
    tail = (lower / cap) ** shape
    if abs(shape - 1.0) < 1e-9:
        return lower * math.log(cap / lower) / (1.0 - tail)
    return (
        shape
        * lower**shape
        * (cap ** (1.0 - shape) - lower ** (1.0 - shape))
        / ((1.0 - shape) * (1.0 - tail))
    )


def large_employment_mean_range(
    lower: float = 250.0, cap: float = LARGE_EMPLOYMENT_CAP
) -> tuple[float, float]:
    """The 250+ means the Pareto draw can hit exactly, ``(smallest, largest)``.

    They are the truncated means at the ends of
    ``LARGE_EMPLOYMENT_SHAPE_BRACKET``; a mean outside them would make the
    bisection saturate at a bracket end and miss the mean silently.
    """

    low_shape, high_shape = LARGE_EMPLOYMENT_SHAPE_BRACKET
    return (
        _large_employment_truncated_mean(high_shape, lower, cap),
        _large_employment_truncated_mean(low_shape, lower, cap),
    )


def large_employment_pareto_shape(
    mean: float = LARGE_EMPLOYMENT_MEAN,
    lower: float = 250.0,
    cap: float = LARGE_EMPLOYMENT_CAP,
) -> float:
    """Shape of the Pareto on ``[lower, cap]`` whose mean is ``mean``.

    The truncated mean falls as the shape rises, so bisection on
    ``LARGE_EMPLOYMENT_SHAPE_BRACKET`` converges; the shape-one case is its
    logarithmic limit. A mean outside ``large_employment_mean_range`` raises
    ``ValueError`` rather than saturating at a bracket end.
    """

    if not lower < mean < cap:
        raise ValueError(
            f"Large-firm mean {mean!r} must lie strictly inside ({lower}, {cap})."
        )
    smallest, largest = large_employment_mean_range(lower, cap)
    if not smallest <= mean <= largest:
        raise ValueError(
            f"Large-firm mean {mean!r} is outside [{smallest:,.2f}, "
            f"{largest:,.2f}], the means a Pareto shape in "
            f"{LARGE_EMPLOYMENT_SHAPE_BRACKET} can reach on [{lower}, {cap}]."
        )

    low, high = LARGE_EMPLOYMENT_SHAPE_BRACKET
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _large_employment_truncated_mean(middle, lower, cap) > mean:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def large_employment_slice_means(
    masses: np.ndarray,
    mean: float = LARGE_EMPLOYMENT_MEAN,
    lower: float = 250.0,
    cap: float = LARGE_EMPLOYMENT_CAP,
) -> np.ndarray:
    """Mean headcount of consecutive probability slices of the large-firm draw.

    ``masses`` are the slices' non-negative sizes, lowest quantiles first;
    they are rescaled to sum to one. Each slice gets the mean of the
    truncated Pareto on ``[lower, cap]`` with mean ``mean`` over its quantile
    range, so the mass-weighted mean of the result is ``mean`` exactly. A
    zero-mass slice gets the quantile at its position.
    """

    masses = np.asarray(masses, dtype=np.float64)
    if masses.size == 0:
        return np.empty(0, dtype=np.float64)
    if np.any(masses < 0) or not masses.sum() > 0:
        raise ValueError("Large-firm slice masses must be non-negative, not all zero.")
    shape = large_employment_pareto_shape(mean, lower, cap)
    span = 1.0 - (lower / cap) ** shape
    exponent = 1.0 - 1.0 / shape
    cumulative = np.concatenate([[0.0], np.cumsum(masses)]) / masses.sum()
    cumulative[-1] = 1.0
    edges = 1.0 - cumulative * span
    if abs(exponent) < 1e-12:
        integral = np.log(edges) / span
    else:
        integral = edges**exponent / (exponent * span)
    width = np.diff(cumulative)
    quantile = lower * edges[:-1] ** (-1.0 / shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = lower * (integral[:-1] - integral[1:]) / width
    return np.where(width > 0, means, quantile)


NEGATIVE_LIABILITY_SECTORS = {
    1,
    3,
    6,
    7,
    9,
    10,
    24,
    30,
    36,
    37,
    49,
    50,
    51,
    60,
    64,
    79,
    84,
}
HIGH_LIABILITY_SECTORS = {11, 12, 69, 70, 78}

#: Input/output ratio draw (microcosm#258, paper #21): ``offset + Beta(a, b) *
#: scale`` plus sector shifts and noise, clamped to ``[lower, upper]``. The
#: mean ratio is about 0.6, so value added is about 40% of turnover and always
#: strictly positive.
INPUT_RATIO_BETA_A = 4.0
INPUT_RATIO_BETA_B = 2.0
INPUT_RATIO_OFFSET = 0.2
INPUT_RATIO_SCALE = 0.6
INPUT_RATIO_LOWER = 0.1
INPUT_RATIO_UPPER = 0.95

#: The paper's ``--fast`` sampling preset (``firm_microsim.generate``).
FAST_SAMPLE_WINDOW_FRACTION = 0.30
FAST_SAMPLE_TAIL_FRACTION = 0.05

#: Upper edges of the ONS employment bands below the open top band, in
#: :data:`EMPLOYMENT_BANDS` order, for vectorised band lookup.
_EMPLOYMENT_BAND_UPPER_EDGES: np.ndarray = np.asarray(
    [upper for _, upper, _ in ONS_EMPLOYMENT_BANDS.values()][:-1],
    dtype=np.float64,
)

AXIOM_UK_VAT_NET_LIABILITY_OUTPUT = (
    "uk:statutes/ukpga/1994/23/25#net_vat_liability_after_input_tax_credit"
)
AXIOM_UK_VAT_TAXABLE_PERSON_INPUT = "uk:statutes/ukpga/1994/23/24#input.taxable_person"
AXIOM_UK_VAT_SUPPLIES_INPUT = (
    "uk:statutes/ukpga/1994/23/24#input.standard_rated_taxable_supplies_value"
)
AXIOM_UK_VAT_PURCHASES_INPUT = "uk:statutes/ukpga/1994/23/24#input.standard_rated_deductible_business_purchase_value"
AXIOM_UK_VAT_ALLOWABLE_PROPORTION_INPUT = (
    "uk:statutes/ukpga/1994/23/26#input.allowable_input_tax_proportion"
)


class UKFirmVATRuleEvaluator(Protocol):
    """Evaluate VAT liability for candidate firms before weight optimization."""

    def net_liability_k(
        self,
        *,
        turnover_k: np.ndarray,
        input_cost_k: np.ndarray,
        vat_registered: np.ndarray,
        data_vintage: str,
    ) -> np.ndarray:
        """Return net VAT liability in thousands of pounds."""


@dataclass(frozen=True)
class AxiomVATRuleEvaluator:
    """Run UK VAT liability through a compiled Axiom RuleSpec artifact."""

    artifact_path: str | Path
    engine_binary: str | Path = "axiom-rules-engine"
    allowable_input_tax_proportion: float = 1.0
    output_id: str = AXIOM_UK_VAT_NET_LIABILITY_OUTPUT

    def net_liability_k(
        self,
        *,
        turnover_k: np.ndarray,
        input_cost_k: np.ndarray,
        vat_registered: np.ndarray,
        data_vintage: str,
    ) -> np.ndarray:
        turnover = np.asarray(turnover_k, dtype=np.float64)
        input_cost = np.asarray(input_cost_k, dtype=np.float64)
        registered = np.asarray(vat_registered, dtype=bool)
        if turnover.shape != input_cost.shape or turnover.shape != registered.shape:
            raise ValueError("VAT rule inputs must have matching shapes.")

        period = _vat_rule_period(data_vintage)
        interval = {"start": period["start"], "end": period["end"]}
        request_inputs: list[dict[str, object]] = []
        queries: list[dict[str, object]] = []
        for index, (sales_k, purchases_k, taxable) in enumerate(
            zip(turnover, input_cost, registered, strict=True),
        ):
            entity_id = f"firm:{index + 1}"
            request_inputs.extend(
                [
                    _axiom_input_record(
                        AXIOM_UK_VAT_TAXABLE_PERSON_INPUT,
                        entity_id,
                        interval,
                        {"kind": "bool", "value": bool(taxable)},
                    ),
                    _axiom_input_record(
                        AXIOM_UK_VAT_SUPPLIES_INPUT,
                        entity_id,
                        interval,
                        {"kind": "decimal", "value": _decimal_string(sales_k * 1000.0)},
                    ),
                    _axiom_input_record(
                        AXIOM_UK_VAT_PURCHASES_INPUT,
                        entity_id,
                        interval,
                        {
                            "kind": "decimal",
                            "value": _decimal_string(purchases_k * 1000.0),
                        },
                    ),
                    _axiom_input_record(
                        AXIOM_UK_VAT_ALLOWABLE_PROPORTION_INPUT,
                        entity_id,
                        interval,
                        {
                            "kind": "decimal",
                            "value": _decimal_string(
                                self.allowable_input_tax_proportion
                            ),
                        },
                    ),
                ]
            )
            queries.append(
                {
                    "entity_id": entity_id,
                    "period": period,
                    "outputs": [self.output_id],
                }
            )

        request = {
            "mode": "fast",
            "dataset": {"inputs": request_inputs, "relations": []},
            "queries": queries,
        }
        process = subprocess.run(
            [
                str(self.engine_binary),
                "run-compiled",
                "--artifact",
                str(self.artifact_path),
            ],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=False,
        )
        if process.returncode != 0:
            stderr = process.stderr.strip() or "Axiom VAT rule evaluation failed"
            raise RuntimeError(stderr)

        response = json.loads(process.stdout)
        values = []
        for result in response["results"]:
            output = result["outputs"][self.output_id]
            values.append(float(output["value"]["value"]) / 1000.0)
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class UKFirmGenerationConfig:
    """Configuration for the experimental UK firm generator.

    VAT liability comes from ``firm_rules`` (a
    :class:`~microcosm.build.uk_runtime.firm_rules.UKFirmRules`). When it is
    absent, a legacy ``vat_rule_evaluator`` is wrapped in
    :class:`~microcosm.build.uk_runtime.firm_rules.LegacyVATEvaluatorRules`.

    Stratified thinning runs when ``sample_window_fraction`` or
    ``sample_tail_fraction`` is below one: rows with turnover inside
    ``[sample_window_lo_k, sample_window_hi_k]`` are kept at the window
    fraction, others at the tail fraction, with at least ``sample_cell_floor``
    rows per sector x HMRC band x window-side stratum. The defaults keep every
    row; :meth:`fast` is the paper's fast-iteration preset.

    ``large_employment_mean`` (headcount) replaces the paper's 250+ draw,
    clamped at 2,000 staff, with a truncated Pareto on
    ``[250, LARGE_EMPLOYMENT_CAP]`` with that base-weighted mean over the
    ONS-frame rows; :data:`LARGE_EMPLOYMENT_MEAN` is DBT's private-sector
    mean. Out-of-frame rows (the HMRC zero-turnover traders) then draw a
    headcount below 250, so every 250+ firm is a frame firm. ``None`` (the
    default) keeps the paper's draw, in which they can reach 250+.

    ``bpe_turnover`` (a ``firm_headcount.UKFirmBPESource``; ``None``, the
    default, keeps the paper's draws) switches on the opt-in BPE turnover
    fix (``firm_bpe_turnover.py``): the open ONS turnover band takes a
    truncated Pareto tail pinned to DBT BPE 2025 total turnover instead of
    the log-uniform draw capped at 50 million pounds, and frame rows'
    employment is re-paired with turnover inside each SIC division by a
    Gaussian copula whose ``rho`` is fitted to BPE turnover by employment
    band. Both act before the solve and draw nothing from the generator's
    stream; on an unthinned build its draws are unchanged, while a thinned
    build stratifies on the new turnover and so keeps different rows.

    ``vat_threshold`` defaults to the ``data_vintage``'s threshold, resolved
    once at construction. ``dataclasses.replace(config, data_vintage=...)``
    therefore keeps the old vintage's threshold (it is passed back in as an
    explicit value); use :meth:`with_vintage` to switch vintage and threshold
    together. An explicit ``vat_threshold`` that differs from the vintage's
    (a counterfactual threshold) is allowed.
    """

    data_vintage: str = DEFAULT_VINTAGE
    vat_threshold: float | None = None
    seed: int = 42
    device: str = "cpu"
    n_iterations: int = 1_000
    learning_rate: float = 0.01
    target_loss_cap: float = 10.0
    max_weight_ratio: float | None = None
    conserve_mass: bool = False
    turnover_importance: float = 5.0
    population_importance: float = 1.0
    sector_importance: float = 1.0
    employment_importance: float = 1.0
    vat_liability_sector_importance: float = 1.0
    vat_liability_band_importance: float = 2.0
    calibrate_vat_liability_sector: bool = False
    input_files: dict[str, str] = field(default_factory=lambda: dict(INPUT_FILES))
    vat_rule_evaluator: UKFirmVATRuleEvaluator | None = None
    sample_tail_fraction: float = 1.0
    sample_window_fraction: float = 1.0
    sample_window_lo_k: float = 15.0
    sample_window_hi_k: float = 150.0
    sample_cell_floor: int = 40
    firm_rules: UKFirmRules | None = None
    large_employment_mean: float | None = None
    bpe_turnover: Any = None

    @classmethod
    def fast(cls, **overrides: Any) -> UKFirmGenerationConfig:
        """The paper's fast-iteration preset: window 0.30, tails 0.05.

        Mirrors ``firm_microsim.generate(fast=True)``. Base weights keep every
        calibration target a true population total, so a fast build solves
        the same problem around a rescaled prior.
        """

        values: dict[str, Any] = {
            "sample_window_fraction": FAST_SAMPLE_WINDOW_FRACTION,
            "sample_tail_fraction": FAST_SAMPLE_TAIL_FRACTION,
        }
        values.update(overrides)
        return cls(**values)

    def with_vintage(self, data_vintage: str) -> UKFirmGenerationConfig:
        """This configuration for ``data_vintage`` at that vintage's threshold.

        Every other field is kept. Unlike ``dataclasses.replace(config,
        data_vintage=...)``, the VAT threshold follows the new vintage.
        """

        if data_vintage not in VINTAGES:
            raise ValueError(
                f"Unknown UK firm vintage {data_vintage!r}; "
                f"choose from {sorted(VINTAGES)}."
            )
        return replace(
            self,
            data_vintage=data_vintage,
            vat_threshold=float(VINTAGES[data_vintage]["threshold"]),
        )

    @property
    def thinned(self) -> bool:
        """Whether this configuration thins the candidate draw."""

        return self.sample_window_fraction < 1.0 or self.sample_tail_fraction < 1.0

    def __post_init__(self) -> None:
        if self.data_vintage not in VINTAGES:
            raise ValueError(
                f"Unknown UK firm vintage {self.data_vintage!r}; "
                f"choose from {sorted(VINTAGES)}."
            )
        if self.vat_threshold is None:
            object.__setattr__(
                self,
                "vat_threshold",
                float(VINTAGES[self.data_vintage]["threshold"]),
            )
        if self.vat_threshold <= 0:
            raise ValueError("vat_threshold must be positive.")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")
        for name in ("sample_tail_fraction", "sample_window_fraction"):
            fraction = float(getattr(self, name))
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1], got {fraction}.")
        if self.sample_window_lo_k > self.sample_window_hi_k:
            raise ValueError("sample_window_lo_k must not exceed sample_window_hi_k.")
        if self.sample_cell_floor < 0:
            raise ValueError("sample_cell_floor must be non-negative.")
        if self.large_employment_mean is not None:
            smallest, largest = large_employment_mean_range()
            if not smallest <= float(self.large_employment_mean) <= largest:
                raise ValueError(
                    f"large_employment_mean must lie in [{smallest:,.2f}, "
                    f"{largest:,.2f}], the means the Pareto draw can hit "
                    "exactly."
                )


def _vat_rule_period(data_vintage: str) -> dict[str, str]:
    start_year = int(data_vintage.split("-", 1)[0])
    return {
        "period_kind": "tax_year",
        "start": f"{start_year}-04-06",
        "end": f"{start_year + 1}-04-05",
    }


def _axiom_input_record(
    name: str,
    entity_id: str,
    interval: Mapping[str, str],
    value: Mapping[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "entity": "Firm",
        "entity_id": entity_id,
        "interval": dict(interval),
        "value": dict(value),
    }


def _decimal_string(value: float) -> str:
    return format(float(value), ".12g")


@dataclass(frozen=True)
class UKFirmSourceData:
    """Processed ONS/HMRC source tables used by the firm generator."""

    ons_turnover: pd.DataFrame
    ons_employment: pd.DataFrame
    hmrc_population_band: pd.DataFrame
    hmrc_population_sector: pd.DataFrame
    hmrc_liability_band: pd.DataFrame
    hmrc_liability_sector: pd.DataFrame
    ons_total: int
    hmrc_bands: dict[str, float]
    vat_liability_bands: dict[str, float]


@dataclass(frozen=True)
class UKFirmTargetLayout:
    """Target row layout in the generated calibration matrix.

    Rows run: HMRC registered-trader counts for all eight HMRC turnover bands
    (``Negative_or_Zero`` included, since its traders enter the solve), the
    ONS frame population, HMRC registered traders by sector, ONS employment
    bands, optional net VAT by sector, then net VAT by calibrated band.
    """

    n_sectors: int
    n_vat_sectors: int

    @property
    def n_turnover(self) -> int:
        return len(HMRC_BAND_COLUMNS)

    @property
    def n_population(self) -> int:
        return 1

    @property
    def n_employment(self) -> int:
        return len(EMPLOYMENT_BANDS)

    @property
    def n_vat_bands(self) -> int:
        return len(VAT_LIABILITY_BANDS_CALIBRATED)

    @property
    def turnover_start(self) -> int:
        return 0

    @property
    def population_start(self) -> int:
        return self.turnover_start + self.n_turnover

    @property
    def sector_start(self) -> int:
        return self.population_start + self.n_population

    @property
    def employment_start(self) -> int:
        return self.sector_start + self.n_sectors

    @property
    def vat_sector_start(self) -> int:
        return self.employment_start + self.n_employment

    @property
    def vat_band_start(self) -> int:
        return self.vat_sector_start + self.n_vat_sectors

    @property
    def n_targets(self) -> int:
        return self.vat_band_start + self.n_vat_bands


@dataclass(frozen=True)
class UKFirmValidationReport:
    """Accuracy scores for the experimental generated firm population."""

    hmrc_bands: float
    ons_population: float
    employment: float
    sector: float
    vat_liability_sector: float
    vat_liability_band: float
    total_population: float

    @property
    def overall(self) -> float:
        """Mean accuracy across calibrated dimensions."""

        return (
            self.hmrc_bands
            + self.ons_population
            + self.employment
            + self.sector
            + self.vat_liability_band
        ) / 5.0


@dataclass(frozen=True)
class UKFirmCalibrationResult:
    """Core Microcosm calibration result for every generated row.

    ``weights`` and ``initial_weights`` cover the ONS-frame rows and the
    appended HMRC negative/zero-turnover traders, in output row order.
    ``initial_weights`` are the stratified-thinning base weights (one for
    every row in an unthinned build).
    """

    weights: Tensor
    loss_trajectory: np.ndarray
    initial_loss: float
    final_loss: float
    target_loss_weights: np.ndarray
    target_loss_scales: np.ndarray
    initial_weights: np.ndarray | None = None


@dataclass(frozen=True)
class UKFirmGenerationResult:
    """Generated firm rows plus validation and calibration diagnostics."""

    firms: pd.DataFrame
    validation: UKFirmValidationReport
    calibration: UKFirmCalibrationResult
    target_diagnostics: pd.DataFrame
    config: UKFirmGenerationConfig
    bpe_turnover: Mapping[str, Any] | None = None


def read_uk_firm_source_data(
    processed_dir: str | Path,
    *,
    input_files: dict[str, str] | None = None,
) -> UKFirmSourceData:
    """Read processed ONS/HMRC firm input tables from ``processed_dir``."""

    root = Path(processed_dir)
    files = input_files or INPUT_FILES
    frames = {key: pd.read_csv(root / filename) for key, filename in files.items()}
    return uk_firm_source_data_from_frames(**frames)


def uk_firm_source_data_from_frames(
    *,
    ons_turnover: pd.DataFrame,
    ons_employment: pd.DataFrame,
    hmrc_population_band: pd.DataFrame,
    hmrc_population_sector: pd.DataFrame,
    hmrc_liability_band: pd.DataFrame,
    hmrc_liability_sector: pd.DataFrame,
) -> UKFirmSourceData:
    """Build source-data metadata from processed input frames."""

    hmrc_bands = _latest_hmrc_band_targets(hmrc_population_band)
    vat_liability_bands = _latest_vat_liability_bands(hmrc_liability_band)
    return UKFirmSourceData(
        ons_turnover=ons_turnover.copy(),
        ons_employment=ons_employment.copy(),
        hmrc_population_band=hmrc_population_band.copy(),
        hmrc_population_sector=hmrc_population_sector.copy(),
        hmrc_liability_band=hmrc_liability_band.copy(),
        hmrc_liability_sector=hmrc_liability_sector.copy(),
        ons_total=_extract_ons_total(ons_turnover),
        hmrc_bands=hmrc_bands,
        vat_liability_bands=vat_liability_bands,
    )


def uk_firm_target_profile_from_mapping(
    raw: Mapping[str, Any],
) -> UKFirmLedgerTargetProfile:
    """Extract UK firm record-set selectors from a Ledger target profile."""

    target_rows = raw.get("targets")
    if not isinstance(target_rows, Iterable) or isinstance(
        target_rows, str | bytes | Mapping
    ):
        raise ValueError("UK firm target profile must contain a targets list.")

    targets_by_id: dict[str, Mapping[str, Any]] = {}
    for row in target_rows:
        if not isinstance(row, Mapping):
            raise ValueError("UK firm target profile targets must be objects.")
        target_id = row.get("target_id")
        if isinstance(target_id, str):
            targets_by_id[target_id] = row

    record_sets: dict[str, str] = {}
    for logical_key, target_id in UK_FIRM_TARGET_IDS.items():
        target = targets_by_id.get(target_id)
        if target is None:
            raise ValueError(f"UK firm target profile missing target {target_id!r}.")

        selector = target.get("ledger_selector")
        if not isinstance(selector, Mapping):
            raise ValueError(
                f"UK firm target {target_id!r} must define ledger_selector."
            )

        record_set_id = selector.get("record_set_id")
        if not isinstance(record_set_id, str) or not record_set_id:
            raise ValueError(
                f"UK firm target {target_id!r} must define "
                "ledger_selector.record_set_id."
            )

        record_sets[_UK_FIRM_TARGET_PROFILE_FIELDS[logical_key]] = record_set_id

    return UKFirmLedgerTargetProfile(**record_sets)


def _coerce_uk_firm_target_profile(
    target_profile: UKFirmLedgerTargetProfile | Mapping[str, Any] | None,
) -> UKFirmLedgerTargetProfile:
    if target_profile is None:
        return uk_firm_target_profile_from_mapping(load_uk_firms_contract())
    if isinstance(target_profile, UKFirmLedgerTargetProfile):
        return target_profile
    return uk_firm_target_profile_from_mapping(target_profile)


def uk_firm_source_data_from_ledger_facts(
    facts: Iterable[Mapping[str, Any] | object],
    *,
    data_vintage: str = "2024-25",
    target_profile: UKFirmLedgerTargetProfile | Mapping[str, Any] | None = None,
) -> UKFirmSourceData:
    """Build UK firm inputs from Ledger consumer facts.

    The accepted rows are the Ledger/Arch consumer-fact shape emitted from the
    ``uk_firms`` target profile. Ledger values are the calibration target source;
    this adapter only reshapes them into the generator's temporary matrix
    contract while firm support is being migrated.
    """

    profile = _coerce_uk_firm_target_profile(target_profile)
    fact_rows = tuple(facts)
    ons_turnover = _ledger_ons_sic_band_matrix(
        fact_rows,
        record_set_id=profile.ons_sic_turnover_record_set,
        band_map=LEDGER_ONS_TURNOVER_BANDS,
        band_dimension="uk.firm.turnover_band",
        table_name="ONS SIC turnover",
    )
    ons_employment = _ledger_ons_sic_band_matrix(
        fact_rows,
        record_set_id=profile.ons_sic_employment_record_set,
        band_map=LEDGER_ONS_EMPLOYMENT_BANDS,
        band_dimension="uk.firm.employment_band",
        table_name="ONS SIC employment",
    )
    hmrc_population_values = _ledger_values_by_band(
        fact_rows,
        record_set_id=profile.hmrc_population_band_record_set,
        measure_id="vat_registered_trader_count",
        band_map=LEDGER_HMRC_BANDS,
    )
    hmrc_liability_values_gbp = _ledger_values_by_band(
        fact_rows,
        record_set_id=profile.hmrc_liability_band_record_set,
        measure_id="net_vat_liability",
        band_map=LEDGER_HMRC_BANDS,
    )
    hmrc_liability_values_m = {
        band: value / 1_000_000.0 for band, value in hmrc_liability_values_gbp.items()
    }
    hmrc_population_band = pd.DataFrame(
        [{"Financial_Year": data_vintage, **hmrc_population_values}]
    )
    hmrc_population_sector = _ledger_sic_series(
        fact_rows,
        record_set_id=profile.hmrc_population_sic_record_set,
        measure_id="vat_registered_trader_count",
        data_vintage=data_vintage,
        value_scale=1.0,
    )
    hmrc_liability_band = pd.DataFrame(
        [{"Financial_Year": data_vintage, **hmrc_liability_values_m}]
    )
    hmrc_liability_sector = _ledger_sic_series(
        fact_rows,
        record_set_id=profile.hmrc_liability_sic_record_set,
        measure_id="net_vat_liability",
        data_vintage=data_vintage,
        value_scale=1 / 1_000_000.0,
    )
    return uk_firm_source_data_from_frames(
        ons_turnover=ons_turnover,
        ons_employment=ons_employment,
        hmrc_population_band=hmrc_population_band,
        hmrc_population_sector=hmrc_population_sector,
        hmrc_liability_band=hmrc_liability_band,
        hmrc_liability_sector=hmrc_liability_sector,
    )


def generate_uk_firm_population(
    data: UKFirmSourceData,
    config: UKFirmGenerationConfig | None = None,
    *,
    vintage: str | None = None,
    threshold: float | None = None,
    seed: int | None = None,
) -> UKFirmGenerationResult:
    """Generate an experimental synthetic UK firm population.

    Args:
        data: Ledger-materialized UK firm targets reshaped by
            :func:`uk_firm_source_data_from_ledger_facts`, or processed
            ONS/HMRC migration-harness inputs.
        config: Generator settings. Defaults to :class:`UKFirmGenerationConfig`.
        vintage: Optional data-vintage override, which also selects the
            vintage default threshold unless ``threshold`` is supplied.
        threshold: Optional VAT threshold override in thousands of pounds.
        seed: Optional random-seed override.

    Returns:
        A :class:`UKFirmGenerationResult` with firm rows, validation, and
        per-target calibration diagnostics.
    """

    cfg = _resolve_config(config, vintage=vintage, threshold=threshold, seed=seed)
    rules = resolve_uk_firm_rules(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    hmrc_bands = _hmrc_band_targets_for_vintage(
        data.hmrc_population_band,
        cfg.data_vintage,
    )
    hmrc_counts = np.asarray(
        [hmrc_bands[band] for band in HMRC_BAND_COLUMNS],
        dtype=np.float64,
    )
    threshold_k = float(cfg.vat_threshold)

    frame_sic, frame_turnover = generate_base_firms(data.ons_turnover, cfg.device)
    if len(frame_sic) == 0:
        raise ValueError("UK firm generation requires at least one ONS firm row.")
    bpe_turnover: dict[str, Any] | None = None
    if cfg.bpe_turnover is not None:
        frame_turnover, bpe_turnover = _bpe_open_band_tail(
            frame_sic, frame_turnover, data, hmrc_bands, cfg
        )
    frame_input = generate_input_values(frame_turnover, frame_sic, cfg.device)
    frame_base_weights = np.ones(len(frame_sic), dtype=np.float64)
    if cfg.thinned:
        keep_idx, frame_base_weights = stratified_thin(
            frame_turnover.detach().cpu().numpy(),
            frame_sic.detach().cpu().numpy(),
            map_to_hmrc_band_indices(frame_turnover, threshold_k).cpu().numpy(),
            window_lo_k=cfg.sample_window_lo_k,
            window_hi_k=cfg.sample_window_hi_k,
            window_fraction=cfg.sample_window_fraction,
            tail_fraction=cfg.sample_tail_fraction,
            cell_floor=cfg.sample_cell_floor,
            seed=cfg.seed,
        )
        keep = torch.as_tensor(keep_idx, dtype=torch.long, device=cfg.device)
        frame_sic = frame_sic[keep]
        frame_turnover = frame_turnover[keep]
        frame_input = frame_input[keep]

    # HMRC negative/zero-turnover traders join BEFORE calibration (paper
    # issue #37): every target row sees the same rows, they sit outside the
    # ONS frame, and they count toward the HMRC registered-trader rows.
    sic_codes, turnover, input_values, base_weights, frame_mask = (
        _add_zero_turnover_firms(
            frame_sic,
            frame_turnover,
            frame_input,
            frame_base_weights,
            hmrc_bands,
            cfg.device,
        )
    )
    # Employment is drawn once, for every row, and kept: the employment
    # target rows and the output describe the same firms.
    employment = assign_employment(
        sic_codes,
        data.ons_employment,
        cfg.device,
        weights=base_weights,
        large_employment_mean=cfg.large_employment_mean,
        in_frame=frame_mask,
    )
    employment_np = employment.detach().cpu().numpy()
    band_idx = map_to_hmrc_band_indices(turnover, threshold_k).cpu().numpy()
    firm_ids = np.arange(1, len(sic_codes) + 1, dtype=np.int64)
    sic_np = sic_codes.detach().cpu().numpy().astype(np.int64)
    turnover_np = turnover.detach().cpu().numpy().astype(np.float64)
    if cfg.bpe_turnover is not None and bpe_turnover is not None:
        in_frame_np = _as_numpy(frame_mask, bool)
        fit = fit_uk_firm_employment_turnover_rho(
            sic_np,
            turnover_np,
            employment_np,
            in_frame_np,
            _as_numpy(base_weights, np.float64),
            cfg.bpe_turnover,
            seed=cfg.seed,
            firm_ids=firm_ids,
        )
        employment_np = repair_uk_firm_employment_turnover(
            sic_np,
            turnover_np,
            employment_np,
            in_frame_np,
            rho=fit.rho,
            seed=cfg.seed,
            firm_ids=firm_ids,
        )
        bpe_turnover["rho"] = fit.rho
        bpe_turnover["rho_sse"] = fit.sse
        bpe_turnover["rho_at_bound"] = fit.at_bound
        bpe_turnover["rho_fit"] = fit.table.to_dict(orient="records")
    employment_band_indices = employment_band_index_array(employment_np)
    input_np = input_values.detach().cpu().numpy().astype(np.float64)

    liability_if_registered_k = evaluate_vat_liability_if_registered_k(
        rules,
        pd.DataFrame(
            {
                "firm_id": firm_ids,
                "sic_code": sic_np,
                "annual_turnover_k": turnover_np,
                "annual_input_k": input_np,
                "employment": employment_np,
            }
        ),
        cfg.data_vintage,
    )
    propensity = registration_propensity(
        band_idx,
        frame_mask,
        base_weights,
        hmrc_counts,
    )

    target_matrix, target_values, layout, target_names = build_firm_target_matrix(
        cfg,
        turnover_np,
        sic_np,
        employment_band_indices,
        data,
        propensity=propensity,
        frame_mask=frame_mask,
        vat_liability_if_registered_k=liability_if_registered_k,
    )
    calibration = solve_firm_weights(
        cfg,
        target_matrix,
        target_values,
        layout,
        initial_weights=base_weights,
    )
    weights_np = calibration.weights.detach().cpu().numpy().astype(np.float64)
    vat_scope, vat_registered = assign_vat_scope_and_registration(
        band_idx,
        frame_mask,
        weights_np,
        hmrc_counts,
        seed=cfg.seed,
    )
    firms = _assemble_firm_rows(
        firm_ids,
        sic_np,
        turnover_np,
        input_np,
        np.where(vat_registered, liability_if_registered_k, 0.0),
        employment_np,
        weights_np,
        base_weights,
        frame_mask,
        vat_scope,
        vat_registered,
        cfg,
    )
    validation = validate_uk_firm_population(firms, data, cfg)
    diagnostics = target_diagnostics(
        target_matrix,
        target_values,
        weights_np,
        target_names,
        initial_weights=base_weights,
    )
    return UKFirmGenerationResult(
        firms=firms,
        validation=validation,
        calibration=calibration,
        target_diagnostics=diagnostics,
        config=cfg,
        bpe_turnover=bpe_turnover,
    )


def _bpe_open_band_tail(
    frame_sic: Tensor,
    frame_turnover: Tensor,
    data: UKFirmSourceData,
    hmrc_bands: Mapping[str, float],
    cfg: UKFirmGenerationConfig,
) -> tuple[Tensor, dict[str, Any]]:
    """Frame turnover with the open band redrawn on the BPE-pinned tail."""

    sic = frame_sic.detach().cpu().numpy().astype(np.int64)
    turnover = frame_turnover.detach().cpu().numpy().astype(np.float64)
    open_band = turnover >= OPEN_BAND_LOWER_K
    tail = uk_firm_open_band_tail_parameters(
        data.ons_turnover,
        hmrc_bands,
        cfg.bpe_turnover,
        sic=sic,
        turnover_k=turnover,
        in_frame=np.ones(len(sic), dtype=bool),
        open_band=open_band,
    )
    uniforms = uk_firm_open_band_uniforms(turnover[open_band])
    section_k = np.isin(sic[open_band], (64, 65, 66))
    turnover[open_band] = draw_uk_firm_open_band_turnover(
        uniforms, tail.alpha, tail.cap_k, strata=section_k
    )
    record = {
        "alpha": tail.alpha,
        "cap_k": tail.cap_k,
        "target_mean_k": tail.target_mean_k,
        "p_over_10m": tail.p_over_10m,
        "model_p_over_10m": tail.model_p_over_10m,
        "open_band_rows": int(open_band.sum()),
    }
    return (
        torch.tensor(
            turnover, dtype=frame_turnover.dtype, device=frame_turnover.device
        ),
        record,
    )


def write_uk_firm_population(
    result: UKFirmGenerationResult,
    path: str | Path,
) -> Path:
    """Write generated firm rows to CSV and return the path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.firms.to_csv(output, index=False)
    return output


def generate_base_firms(
    ons_turnover: pd.DataFrame,
    device: str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Draw base firms from ONS sector-by-turnover counts."""

    sic_codes: list[int] = []
    turnovers: list[np.ndarray] = []
    for _, row in ons_turnover.iterrows():
        sic_code = row["SIC Code"]
        if pd.isna(sic_code) or str(sic_code) in {"", "Total"}:
            continue
        sic_int = int(sic_code)
        for band, (lower, upper, _midpoint) in ONS_TURNOVER_BANDS.items():
            if band not in row or pd.isna(row[band]) or row[band] <= 0:
                continue
            count = int(row[band])
            sic_codes.extend([sic_int] * count)
            turnovers.append(
                _draw_band_turnover(
                    count,
                    lower,
                    upper,
                    device,
                    log_uniform=band == ONS_OPEN_TURNOVER_BAND,
                )
                .cpu()
                .numpy()
            )

    turnover_values = (
        np.concatenate(turnovers).astype(np.float32)
        if turnovers
        else np.array([], dtype=np.float32)
    )
    return (
        torch.tensor(sic_codes, dtype=torch.int64, device=device),
        torch.tensor(turnover_values, dtype=torch.float32, device=device),
    )


def generate_input_values(
    turnover_values: Tensor,
    sic_codes: Tensor,
    device: str = "cpu",
) -> Tensor:
    """Draw annual input expenditure in thousands of pounds.

    The input/output ratio is ``0.2 + Beta(4, 2) * 0.6`` (mean about 0.6)
    with sector shifts and noise, clamped to ``[0.1, 0.95]`` so value added is
    always strictly positive (microcosm#258, ported from the paper's #21).
    """

    n_firms = len(turnover_values)
    if n_firms == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    ratios = (
        torch.distributions.Beta(INPUT_RATIO_BETA_A, INPUT_RATIO_BETA_B)
        .sample((n_firms,))
        .to(device)
    )
    scaled = INPUT_RATIO_OFFSET + ratios * INPUT_RATIO_SCALE
    sector_noise = torch.randn(n_firms, device=device) * 0.15
    sic_np = sic_codes.detach().cpu().numpy()
    neg_mask = np.isin(sic_np, list(NEGATIVE_LIABILITY_SECTORS))
    high_mask = np.isin(sic_np, list(HIGH_LIABILITY_SECTORS))
    neg_t = torch.tensor(neg_mask, device=device)
    high_t = torch.tensor(high_mask, device=device)
    scaled = scaled + neg_t.float() * (torch.rand(n_firms, device=device) * 0.3)
    scaled = scaled - high_t.float() * (torch.rand(n_firms, device=device) * 0.2)
    final_ratios = torch.clamp(
        scaled + sector_noise,
        INPUT_RATIO_LOWER,
        INPUT_RATIO_UPPER,
    )
    return torch.where(
        turnover_values > 0,
        turnover_values * final_ratios,
        torch.zeros_like(turnover_values),
    )


def assign_employment(
    sic_codes: Tensor,
    ons_employment: pd.DataFrame,
    device: str = "cpu",
    weights: np.ndarray | Tensor | None = None,
    large_employment_mean: float | None = None,
    in_frame: np.ndarray | Tensor | None = None,
) -> Tensor:
    """Assign firm employment counts conditional on each firm's ONS sector.

    Ported from the paper's ``generate.assign_employment``: each sector's
    rows draw an employment band from that sector's ONS band shares (the
    national shares when the sector has no ONS employment row or no firms),
    then a count inside the band. Draws use torch's global generator.

    By default the open 250+ band keeps the paper's lognormal draw clamped
    to ``[250, 2000]``. With ``large_employment_mean`` it is not clamped: the
    band's rows, ranked by the same lognormal draw (so the generator's
    stream is unchanged), take the means of consecutive slices of a
    truncated Pareto on ``[250, LARGE_EMPLOYMENT_CAP]`` with that mean, each
    slice's probability proportional to the row's ``weights`` (equal when
    omitted). The band's weighted mean headcount is then that mean, up to
    rounding.

    ``in_frame`` (every row when omitted) marks the ONS-frame rows and is
    read only with ``large_employment_mean``. The published 250+ mean
    describes businesses in the frame, so a row outside it whose band draw
    is 250+ takes no slice of the tail: its lognormal draw instead picks a
    headcount below 250 from its sector's shares of the smaller bands
    (:func:`_employment_below_large_band`). The 250+ band then holds frame
    rows only, and it is their weighted mean that equals the mean. Neither
    option draws from the generator beyond the default's draws.
    """

    sic_codes = torch.as_tensor(sic_codes, dtype=torch.int64, device=device)
    n_firms = len(sic_codes)
    if n_firms == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    sector_rows = ons_employment[
        ~ons_employment["Description"].astype(str).str.contains("Total", na=False)
    ]
    national = torch.tensor(
        [
            float(sector_rows[band].fillna(0).sum()) if band in sector_rows else 0.0
            for band in EMPLOYMENT_BANDS
        ],
        dtype=torch.float32,
        device=device,
    )
    if float(national.sum().item()) <= 0:
        national = torch.ones(len(EMPLOYMENT_BANDS), dtype=torch.float32, device=device)
    national = national / national.sum()
    counts_by_sic: dict[int, Tensor] = {}
    for _, row in sector_rows.iterrows():
        sic = row.get("SIC Code")
        if pd.isna(sic) or str(sic) in {"", "Total"}:
            continue
        counts_by_sic[int(sic)] = torch.tensor(
            [
                float(row[band]) if band in row and pd.notna(row[band]) else 0.0
                for band in EMPLOYMENT_BANDS
            ],
            dtype=torch.float32,
            device=device,
        )

    frame: Tensor | None = None
    if large_employment_mean is not None and in_frame is not None:
        frame = torch.as_tensor(
            _as_numpy(in_frame, bool), dtype=torch.bool, device=device
        )
        if len(frame) != n_firms:
            raise ValueError("in_frame must have one entry per firm.")
    result = torch.empty(n_firms, dtype=torch.float32, device=device)
    large_idx: list[Tensor] = []
    large_scores: list[Tensor] = []
    for sic in torch.unique(sic_codes).tolist():
        idx = torch.where(sic_codes == int(sic))[0]
        counts = counts_by_sic.get(int(sic))
        if counts is None or float(counts.sum().item()) <= 0:
            probs = national
        else:
            probs = counts / counts.sum()
        band_ids = torch.multinomial(probs, len(idx), replacement=True)
        for band_id, band in enumerate(EMPLOYMENT_BANDS):
            out_idx = idx[band_ids == band_id]
            if len(out_idx) == 0:
                continue
            lower, upper, midpoint = ONS_EMPLOYMENT_BANDS[band]
            if band == "250+":
                # The paper's lognormal draw. With a large-employment mean it
                # only ranks the band's rows, keeping the generator's stream
                # unchanged; their headcounts are set after the loop.
                log_mean = torch.log(torch.tensor(float(midpoint), device=device))
                scores = torch.normal(log_mean, 0.8, (len(out_idx),), device=device)
                if large_employment_mean is None:
                    result[out_idx] = torch.clamp(scores.exp(), lower, upper).round()
                elif frame is None:
                    large_idx.append(out_idx)
                    large_scores.append(scores)
                else:
                    inside = frame[out_idx]
                    large_idx.append(out_idx[inside])
                    large_scores.append(scores[inside])
                    outside = out_idx[~inside]
                    if len(outside):
                        result[outside] = torch.as_tensor(
                            _employment_below_large_band(
                                scores[~inside].detach().cpu().numpy(),
                                float(log_mean),
                                0.8,
                                probs.detach().cpu().numpy(),
                                national.detach().cpu().numpy(),
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                continue
            if band == "0-4":
                values = torch.randint(1, 5, (len(out_idx),), device=device).float()
            else:
                values = torch.randint(
                    int(lower), int(upper) + 1, (len(out_idx),), device=device
                ).float()
            result[out_idx] = values
    if large_idx and sum(len(rows) for rows in large_idx):
        # Ranked rows take consecutive weight-sized slices of a truncated
        # Pareto fitted to the published mean, so the band's weighted mean
        # matches the source without sampling noise from the heavy tail.
        lower = ONS_EMPLOYMENT_BANDS["250+"][0]
        rows = torch.cat(large_idx)
        order = torch.argsort(torch.cat(large_scores), stable=True)
        ranked = rows[order]
        if weights is None:
            masses = np.ones(len(ranked), dtype=np.float64)
        else:
            masses = _as_numpy(weights, np.float64)[ranked.cpu().numpy()]
        slices = large_employment_slice_means(
            masses, mean=float(large_employment_mean), lower=lower
        )
        result[ranked] = torch.as_tensor(
            np.round(slices), dtype=torch.float32, device=device
        )
    return result


def _employment_below_large_band(
    scores: np.ndarray,
    log_mean: float,
    log_sd: float,
    shares: np.ndarray,
    fallback_shares: np.ndarray,
) -> np.ndarray:
    """Headcounts below the open 250+ band for rows outside the ONS frame.

    ``scores`` are the rows' normal draws on the log scale (mean
    ``log_mean``, sd ``log_sd``). Each row's quantile of that normal picks
    one of the closed bands by ``shares`` (band shares in
    :data:`EMPLOYMENT_BANDS` order; the open band's share is dropped and
    the rest renormalised) and a whole headcount inside it, so the result
    is deterministic and consumes no random draws. ``fallback_shares``
    serve when ``shares`` has no mass below the open band, and equal shares
    when neither has.
    """

    closed_bands = list(ONS_EMPLOYMENT_BANDS.values())[:-1]
    mass = np.asarray(shares, dtype=np.float64)[: len(closed_bands)]
    if not mass.sum() > 0:
        mass = np.asarray(fallback_shares, dtype=np.float64)[: len(closed_bands)]
    if not mass.sum() > 0:
        mass = np.ones(len(closed_bands), dtype=np.float64)
    mass = mass / mass.sum()
    standard = (np.asarray(scores, dtype=np.float64) - log_mean) / log_sd
    quantile = np.asarray(
        [0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in standard],
        dtype=np.float64,
    )
    upper_edges = np.cumsum(mass)
    last = int(np.flatnonzero(mass > 0)[-1])
    band = np.minimum(np.searchsorted(upper_edges, quantile, side="right"), last)
    lower_edges = np.concatenate([[0.0], upper_edges[:-1]])
    position = np.clip((quantile - lower_edges[band]) / mass[band], 0.0, 1.0)
    lowest = np.asarray([lower for lower, _, _ in closed_bands])[band]
    highest = np.asarray([upper for _, upper, _ in closed_bands])[band]
    return np.minimum(lowest + np.floor(position * (highest - lowest + 1.0)), highest)


def employment_band_index_array(employment: np.ndarray) -> np.ndarray:
    """Vectorised ONS employment-band index (0-6) for employment counts.

    Agrees with :func:`employment_band_name` and the scalar band index.
    """

    values = np.asarray(employment, dtype=np.float64)
    return np.searchsorted(_EMPLOYMENT_BAND_UPPER_EDGES, values, side="left").astype(
        np.int64
    )


def build_firm_target_matrix(
    config: UKFirmGenerationConfig,
    turnover_values: np.ndarray | Tensor,
    sic_codes: np.ndarray | Tensor,
    employment_band_indices: np.ndarray | Tensor,
    data: UKFirmSourceData,
    *,
    propensity: np.ndarray,
    frame_mask: np.ndarray,
    vat_liability_if_registered_k: np.ndarray,
) -> tuple[sparse.csr_array, np.ndarray, UKFirmTargetLayout, tuple[str, ...]]:
    """Construct the firm calibration matrix (float32 CSR) and target vector.

    Two universes (paper issue #37). ONS-frame rows (population, employment
    bands) give frame rows a one and appended HMRC traders a zero. HMRC
    registered-trader rows (counts by band and sector, net VAT by sector and
    calibrated band) give every row its registration ``propensity``; net-VAT
    rows multiply it by the row's liability-if-registered in thousands of
    pounds. The matrix is assembled from COO triples, never densified.
    """

    threshold = float(config.vat_threshold)
    turnover = _as_numpy(turnover_values, np.float64)
    sic = _as_numpy(sic_codes, np.int64)
    employment_band = _as_numpy(employment_band_indices, np.int64)
    frame = np.asarray(frame_mask, dtype=bool)
    prop = np.asarray(propensity, dtype=np.float64)
    liability = np.asarray(vat_liability_if_registered_k, dtype=np.float64)
    n_firms = len(turnover)
    for name, values in (
        ("sic_codes", sic),
        ("employment_band_indices", employment_band),
        ("frame_mask", frame),
        ("propensity", prop),
        ("vat_liability_if_registered_k", liability),
    ):
        if values.shape != (n_firms,):
            raise ValueError(f"{name} has shape {values.shape}; expected ({n_firms},).")

    hmrc_bands = _hmrc_band_targets_for_vintage(
        data.hmrc_population_band,
        config.data_vintage,
    )
    vat_liability_bands = _vat_liability_band_targets_for_vintage(
        data.hmrc_liability_band,
        config.data_vintage,
    )
    sector_targets = hmrc_sector_targets(data, config.data_vintage)
    if config.calibrate_vat_liability_sector:
        vat_sector_rows = data.hmrc_liability_sector[
            data.hmrc_liability_sector["Trade_Sector"] != "Total"
        ].copy()
    else:
        vat_sector_rows = data.hmrc_liability_sector.iloc[0:0].copy()

    layout = UKFirmTargetLayout(
        n_sectors=len(sector_targets),
        n_vat_sectors=len(vat_sector_rows),
    )
    band_indices = map_to_hmrc_band_indices(
        torch.as_tensor(turnover, dtype=torch.float32),
        threshold,
    ).numpy()
    columns = np.arange(n_firms, dtype=np.int64)
    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    value_blocks: list[np.ndarray] = []

    def add(rows: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        row_blocks.append(np.asarray(rows, dtype=np.int64)[mask])
        column_blocks.append(columns[mask])
        value_blocks.append(np.asarray(values, dtype=np.float32)[mask])

    # HMRC registered-trader counts by turnover band: row b for band b.
    add(layout.turnover_start + band_indices, prop != 0, prop)
    target_names = [f"hmrc_vat_population/{band}" for band in HMRC_BAND_COLUMNS]

    # ONS frame population.
    add(
        np.full(n_firms, layout.population_start),
        frame,
        np.ones(n_firms),
    )
    target_names.append("ons_firm_population/total")

    sector_codes = list(sector_targets)
    sector_row = _row_lookup(sic, sector_codes, layout.sector_start)
    add(sector_row, (sector_row >= 0) & (prop != 0), prop)
    target_names.extend(f"hmrc_vat_population_sector/{code}" for code in sector_codes)

    add(
        layout.employment_start + employment_band,
        frame,
        np.ones(n_firms),
    )
    target_names.extend(f"ons_firm_employment/{band}" for band in EMPLOYMENT_BANDS)

    weighted_liability = liability * prop
    vat_sector_codes = [
        int(row["Trade_Sector"]) for _, row in vat_sector_rows.iterrows()
    ]
    vat_sector_row = _row_lookup(sic, vat_sector_codes, layout.vat_sector_start)
    add(
        vat_sector_row,
        (vat_sector_row >= 0) & (weighted_liability != 0),
        weighted_liability,
    )
    target_names.extend(
        f"hmrc_vat_liability_sector/{code}" for code in vat_sector_codes
    )

    band_row = np.full(n_firms, -1, dtype=np.int64)
    for offset, band in enumerate(VAT_LIABILITY_BANDS_CALIBRATED):
        band_row[band_indices == HMRC_BAND_COLUMNS.index(band)] = (
            layout.vat_band_start + offset
        )
    add(band_row, (band_row >= 0) & (weighted_liability != 0), weighted_liability)
    target_names.extend(
        f"hmrc_vat_liability/{band}" for band in VAT_LIABILITY_BANDS_CALIBRATED
    )

    rows = np.concatenate(row_blocks)
    index_dtype = (
        np.int32
        if max(layout.n_targets, n_firms) < np.iinfo(np.int32).max
        else np.int64
    )
    matrix = sparse.csr_array(
        (
            np.concatenate(value_blocks),
            (
                rows.astype(index_dtype),
                np.concatenate(column_blocks).astype(index_dtype),
            ),
        ),
        shape=(layout.n_targets, n_firms),
        dtype=np.float32,
    )

    turnover_targets = [float(hmrc_bands[band]) for band in HMRC_BAND_COLUMNS]
    vat_sector_targets = [
        _vintage_column_value(row, config.data_vintage, "HMRC liability sectors")
        * 1000.0
        for _, row in vat_sector_rows.iterrows()
    ]
    vat_band_targets = [
        float(vat_liability_bands[band]) * 1000.0
        for band in VAT_LIABILITY_BANDS_CALIBRATED
    ]
    targets = np.asarray(
        turnover_targets
        + [float(data.ons_total)]
        + list(sector_targets.values())
        + list(_employment_targets(data.ons_employment).values())
        + vat_sector_targets
        + vat_band_targets,
        dtype=np.float64,
    )
    return matrix, targets, layout, tuple(target_names)


def solve_firm_weights(
    config: UKFirmGenerationConfig,
    target_matrix: sparse.sparray | sparse.spmatrix | Tensor,
    target_values: np.ndarray | Tensor,
    layout: UKFirmTargetLayout,
    *,
    initial_weights: np.ndarray | None = None,
) -> UKFirmCalibrationResult:
    """Solve firm weights with Microcosm's public constraint-system solver.

    ``initial_weights`` default to one per row; stratified builds pass their
    base weights, so the solve starts from (and its result is reported
    against) a draw whose base-weighted totals reproduce the full draw.
    """

    matrix = _as_float32_csr(target_matrix)
    _, n_firms = matrix.shape
    initial = (
        np.ones(n_firms, dtype=np.float64)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=np.float64)
    )
    target_np = _as_numpy(target_values, np.float64)
    loss_weights = _target_loss_weights(layout, config)
    loss_scales = default_target_loss_scales(target_np)

    weights, trajectory = solve_constraint_system(
        matrix,
        target_np,
        initial,
        target_loss_weights=loss_weights,
        target_loss_scales=loss_scales,
        target_loss_cap=config.target_loss_cap,
        mass=CONSERVE_MASS if config.conserve_mass else FREE_MASS,
        max_weight_ratio=config.max_weight_ratio,
        epochs=config.n_iterations,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    initial_estimates = matrix @ initial
    final_estimates = matrix @ weights
    return UKFirmCalibrationResult(
        weights=torch.tensor(weights, dtype=torch.float32, device=config.device),
        loss_trajectory=trajectory,
        initial_loss=relative_error_loss(
            initial_estimates,
            target_np,
            target_loss_weights=loss_weights,
            target_loss_scales=loss_scales,
            target_loss_cap=config.target_loss_cap,
        ),
        final_loss=relative_error_loss(
            final_estimates,
            target_np,
            target_loss_weights=loss_weights,
            target_loss_scales=loss_scales,
            target_loss_cap=config.target_loss_cap,
        ),
        target_loss_weights=loss_weights,
        target_loss_scales=loss_scales,
        initial_weights=initial,
    )


def optimize_firm_weights(
    config: UKFirmGenerationConfig,
    target_matrix: sparse.sparray | sparse.spmatrix | Tensor,
    target_values: np.ndarray | Tensor,
    layout: UKFirmTargetLayout,
    *,
    initial_weights: np.ndarray | None = None,
) -> Tensor:
    """Return core-solved firm weights for callers that only need the vector."""

    return solve_firm_weights(
        config,
        target_matrix,
        target_values,
        layout,
        initial_weights=initial_weights,
    ).weights


def target_diagnostics(
    target_matrix: sparse.sparray | sparse.spmatrix | Tensor,
    target_values: np.ndarray | Tensor,
    weights: np.ndarray | Tensor,
    names: tuple[str, ...],
    *,
    initial_weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return per-target diagnostics for the optimized weight vector.

    ``initial_estimate`` uses ``initial_weights`` (one per row by default).
    """

    matrix = _as_float32_csr(target_matrix)
    weight_np = _as_numpy(weights, np.float64)
    initial = (
        np.ones_like(weight_np)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=np.float64)
    )
    estimates = np.asarray(matrix @ weight_np, dtype=np.float64)
    targets = _as_numpy(target_values, np.float64)
    relative_error = np.divide(
        estimates - targets,
        np.where(np.abs(targets) < 1e-9, 1.0, targets),
    )
    return pd.DataFrame(
        {
            "target_name": names,
            "initial_estimate": np.asarray(matrix @ initial, dtype=np.float64),
            "estimate": estimates,
            "target": targets,
            "relative_error": relative_error,
        }
    )


def hmrc_sector_targets(
    data: UKFirmSourceData,
    data_vintage: str,
) -> dict[int, float]:
    """HMRC registered-trader count targets by sector, on the band universe.

    HMRC sector counts include traders whose turnover band is "Unknown"
    (159,300 in 2024-25), who cannot enter the turnover-band rows. As in the
    paper, the sector counts are scaled so they sum to the eight HMRC band
    counts, allocating the unknown-band traders proportionally; band and
    sector rows then count the same registered universe.
    """

    sector_rows = data.hmrc_population_sector[
        data.hmrc_population_sector["Trade_Sector"] != "Total"
    ]
    raw = {
        int(row["Trade_Sector"]): _vintage_column_value(
            row, data_vintage, "HMRC population sectors"
        )
        for _, row in sector_rows.iterrows()
    }
    hmrc_bands = _hmrc_band_targets_for_vintage(
        data.hmrc_population_band,
        data_vintage,
    )
    band_total = float(sum(hmrc_bands.values()))
    sector_total = float(sum(raw.values()))
    scale = band_total / sector_total if sector_total > 0 else 1.0
    return {sic_code: value * scale for sic_code, value in raw.items()}


def resolve_uk_firm_rules(config: UKFirmGenerationConfig) -> UKFirmRules:
    """The VAT rules a configuration evaluates candidate firms with.

    ``firm_rules`` wins; otherwise a legacy ``vat_rule_evaluator`` is wrapped
    in :class:`~microcosm.build.uk_runtime.firm_rules.LegacyVATEvaluatorRules`.
    """

    if config.firm_rules is not None:
        return config.firm_rules
    if config.vat_rule_evaluator is not None:
        return LegacyVATEvaluatorRules(config.vat_rule_evaluator)
    raise ValueError(
        "UK firm VAT liability requires a VAT rule evaluator. "
        "Pass firm_rules (a UKFirmRules), or AxiomVATRuleEvaluator with a "
        "compiled UK VAT RuleSpec artifact."
    )


def evaluate_vat_liability_if_registered_k(
    rules: UKFirmRules,
    firms: pd.DataFrame,
    data_vintage: str,
) -> np.ndarray:
    """Each candidate firm's VAT liability if registered, in thousands of pounds.

    ``firms`` holds ``firm_id``, ``sic_code``, ``annual_turnover_k``,
    ``annual_input_k`` and ``employment``. The rules see the pre-calibration
    candidate table (policyengine-uk names, pounds) and their outputs are
    aligned back to ``firms`` by ``firm_id``.
    """

    candidates = build_candidate_rules_table(firms)
    outputs = rules.evaluate(candidates, period=rule_period_for_vintage(data_vintage))
    liability_gbp = align_rule_outputs(outputs, firms["firm_id"].to_numpy(np.int64))
    return np.asarray(liability_gbp, dtype=np.float64) / 1000.0


def validate_uk_firm_population(
    firms: pd.DataFrame,
    data: UKFirmSourceData,
    config: UKFirmGenerationConfig,
) -> UKFirmValidationReport:
    """Validate generated firm rows against ONS/HMRC target dimensions."""

    threshold = float(config.vat_threshold)
    hmrc_bands = _hmrc_band_targets_for_vintage(
        data.hmrc_population_band,
        config.data_vintage,
    )
    vat_liability_bands = _vat_liability_band_targets_for_vintage(
        data.hmrc_liability_band,
        config.data_vintage,
    )
    df = firms.copy()
    df["hmrc_band"] = df["annual_turnover_k"].apply(
        lambda value: hmrc_band_name(value, threshold)
    )
    df["weighted_liability_m"] = df["vat_liability_k"] * df["firm_weight"] / 1000.0
    df["sic_numeric"] = df["sic_code"].astype(int)
    registered = df[df["vat_registered"]]
    # Two universes: HMRC margins score registered rows, ONS margins score
    # ONS-frame rows (appended negative/zero-turnover traders are outside it).
    frame = df[df["in_frame"].astype(bool)] if "in_frame" in df else df
    weighted_bands = registered.groupby("hmrc_band")["firm_weight"].sum()
    hmrc_accs = []
    for band_name, target in hmrc_bands.items():
        hmrc_accs.append(_accuracy(float(weighted_bands.get(band_name, 0.0)), target))

    employment_targets = _employment_targets(data.ons_employment)
    weighted_employment = (
        frame.assign(employment_band=frame["employment"].apply(employment_band_name))
        .groupby("employment_band")["firm_weight"]
        .sum()
    )
    employment_accs = [
        _accuracy(float(weighted_employment.get(band, 0.0)), target)
        for band, target in employment_targets.items()
    ]

    weighted_sector = registered.groupby("sic_numeric")["firm_weight"].sum()
    sector_accs = [
        _accuracy(float(weighted_sector.get(sic_code, 0.0)), target)
        for sic_code, target in hmrc_sector_targets(data, config.data_vintage).items()
    ]

    liability_sector_rows = data.hmrc_liability_sector[
        data.hmrc_liability_sector["Trade_Sector"] != "Total"
    ]
    weighted_liability_sector = registered.groupby("sic_numeric")[
        "weighted_liability_m"
    ].sum()
    liability_sector_accs = [
        _accuracy(
            float(weighted_liability_sector.get(int(row["Trade_Sector"]), 0.0)),
            _vintage_column_value(row, config.data_vintage, "HMRC liability sectors"),
        )
        for _, row in liability_sector_rows.iterrows()
    ]

    weighted_liability_band = registered.groupby("hmrc_band")[
        "weighted_liability_m"
    ].sum()
    liability_band_accs = [
        _accuracy(
            float(weighted_liability_band.get(band, 0.0)),
            # HMRC liability targets are in millions; firm rows store thousands.
            float(vat_liability_bands[band]),
        )
        for band in VAT_LIABILITY_BANDS_CALIBRATED
    ]

    return UKFirmValidationReport(
        hmrc_bands=float(np.mean(hmrc_accs)) if hmrc_accs else 0.0,
        ons_population=_accuracy(
            float(frame["firm_weight"].sum()),
            float(data.ons_total),
        ),
        employment=float(np.mean(employment_accs)) if employment_accs else 0.0,
        sector=float(np.mean(sector_accs)) if sector_accs else 0.0,
        vat_liability_sector=(
            float(np.mean(liability_sector_accs)) if liability_sector_accs else 0.0
        ),
        vat_liability_band=(
            float(np.mean(liability_band_accs)) if liability_band_accs else 0.0
        ),
        total_population=float(df["firm_weight"].sum()),
    )


def map_to_hmrc_band_indices(turnover_values: Tensor, threshold: float) -> Tensor:
    """Map turnover values in thousands of pounds to HMRC band indices."""

    band_indices = torch.full_like(turnover_values, 7, dtype=torch.long)
    band_indices = torch.where(turnover_values <= 0, 0, band_indices)
    band_indices = torch.where(
        (turnover_values > 0) & (turnover_values <= threshold),
        1,
        band_indices,
    )
    band_indices = torch.where(
        (turnover_values > threshold) & (turnover_values <= 150),
        2,
        band_indices,
    )
    band_indices = torch.where(
        (turnover_values > 150) & (turnover_values <= 300),
        3,
        band_indices,
    )
    band_indices = torch.where(
        (turnover_values > 300) & (turnover_values <= 500),
        4,
        band_indices,
    )
    band_indices = torch.where(
        (turnover_values > 500) & (turnover_values <= 1000),
        5,
        band_indices,
    )
    band_indices = torch.where(
        (turnover_values > 1000) & (turnover_values <= 10000),
        6,
        band_indices,
    )
    return band_indices


def hmrc_band_name(turnover_k: float, threshold: float) -> str:
    """Map one turnover value in thousands of pounds to an HMRC band name."""

    if turnover_k <= 0:
        return "Negative_or_Zero"
    if turnover_k <= threshold:
        return "£1_to_Threshold"
    if turnover_k <= 150:
        return "£Threshold_to_£150k"
    if turnover_k <= 300:
        return "£150k_to_£300k"
    if turnover_k <= 500:
        return "£300k_to_£500k"
    if turnover_k <= 1000:
        return "£500k_to_£1m"
    if turnover_k <= 10000:
        return "£1m_to_£10m"
    return "Greater_than_£10m"


def employment_band_name(employment: float) -> str:
    """Map employment count to an ONS employment-size band."""

    if employment <= 4:
        return "0-4"
    if employment <= 9:
        return "5-9"
    if employment <= 19:
        return "10-19"
    if employment <= 49:
        return "20-49"
    if employment <= 99:
        return "50-99"
    if employment <= 249:
        return "100-249"
    return "250+"


def _resolve_config(
    config: UKFirmGenerationConfig | None,
    *,
    vintage: str | None,
    threshold: float | None,
    seed: int | None,
) -> UKFirmGenerationConfig:
    cfg = config or UKFirmGenerationConfig()
    overrides: dict[str, object] = {}
    if vintage is not None:
        if vintage not in VINTAGES:
            raise ValueError(
                f"Unknown UK firm vintage {vintage!r}; choose from {sorted(VINTAGES)}."
            )
        overrides["data_vintage"] = vintage
        if threshold is None:
            overrides["vat_threshold"] = VINTAGES[vintage]["threshold"]
    if threshold is not None:
        overrides["vat_threshold"] = float(threshold)
    if seed is not None:
        overrides["seed"] = int(seed)
    return replace(cfg, **overrides) if overrides else cfg


def _draw_band_turnover(
    count: int,
    lower: float,
    upper: float,
    device: str,
    *,
    log_uniform: bool = False,
) -> Tensor:
    """Draw turnover inside the half-open band ``[lower, upper)``.

    Uniform (log-uniform for the open top band), as in the paper generator:
    the maximum-entropy allocation given band membership, which keeps every
    draw inside its source band.
    """

    unit = torch.rand(count, device=device)
    return draw_in_band(
        unit, lower, upper, log_uniform=log_uniform, floor=MIN_DRAWN_TURNOVER_K
    )


def _add_zero_turnover_firms(
    sic_codes: Tensor,
    turnover: Tensor,
    input_values: Tensor,
    base_weights: np.ndarray,
    hmrc_bands: dict[str, float],
    device: str,
) -> tuple[Tensor, Tensor, Tensor, np.ndarray, np.ndarray]:
    """Append HMRC Negative_or_Zero traders, allocated by sector share.

    Called before calibration. Appended rows carry zero turnover and input,
    base weight one and ``frame_mask`` False. Returns ``(sic_codes, turnover,
    input_values, base_weights, frame_mask)``.
    """

    n_frame = len(sic_codes)
    frame_mask = np.ones(n_frame, dtype=bool)
    target = int(round(hmrc_bands["Negative_or_Zero"]))
    add_sics = _allocate_zero_turnover_sics(sic_codes, target) if target > 0 else []
    if not add_sics:
        return sic_codes, turnover, input_values, base_weights, frame_mask
    n_add = len(add_sics)
    zeros = torch.zeros(n_add, dtype=torch.float32, device=device)
    return (
        torch.cat(
            [sic_codes, torch.tensor(add_sics, dtype=torch.int64, device=device)]
        ),
        torch.cat([turnover, zeros]),
        torch.cat([input_values, zeros]),
        np.concatenate([base_weights, np.ones(n_add, dtype=np.float64)]),
        np.concatenate([frame_mask, np.zeros(n_add, dtype=bool)]),
    )


def _allocate_zero_turnover_sics(sic_codes: Tensor, target: int) -> list[int]:
    unique_sics, counts = torch.unique(sic_codes, return_counts=True)
    total = len(sic_codes)
    exact_allocations = target * counts.double() / total
    allocations = torch.floor(exact_allocations).to(torch.int64)
    remainder = target - int(allocations.sum().item())
    if remainder > 0:
        fractional_order = torch.argsort(
            exact_allocations - allocations.double(),
            descending=True,
        )
        allocations[fractional_order[:remainder]] += 1
    add_sics: list[int] = []
    for sic, n_add in zip(unique_sics, allocations, strict=True):
        add_sics.extend([int(sic.item())] * int(n_add.item()))
    return add_sics


def _assemble_firm_rows(
    firm_ids: np.ndarray,
    sic_codes: np.ndarray,
    turnover: np.ndarray,
    input_values: np.ndarray,
    vat_liability: np.ndarray,
    employment: np.ndarray,
    weights: np.ndarray,
    base_weights: np.ndarray,
    frame_mask: np.ndarray,
    vat_scope: np.ndarray,
    vat_registered: np.ndarray,
    config: UKFirmGenerationConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_id": firm_ids,
            "source_firm_id": firm_ids,
            "source_firm_key": [
                f"{config.data_vintage}:experimental_uk_firm:{firm_id}"
                for firm_id in firm_ids
            ],
            "data_vintage": config.data_vintage,
            "sic_code": [str(int(sic)).zfill(5) for sic in sic_codes],
            "annual_turnover_k": np.asarray(turnover, dtype=np.float32),
            "annual_input_k": np.asarray(input_values, dtype=np.float32),
            "vat_liability_k": np.asarray(vat_liability, dtype=np.float32),
            "employment": np.asarray(employment).astype(int),
            "firm_weight": np.asarray(weights, dtype=np.float32),
            "base_weight": np.asarray(base_weights, dtype=np.float64),
            "in_frame": np.asarray(frame_mask, dtype=bool),
            "vat_scope": np.asarray(vat_scope, dtype=bool),
            "vat_registered": np.asarray(vat_registered, dtype=bool),
            "generation_status": "experimental",
        }
    )


def _target_loss_weights(
    layout: UKFirmTargetLayout,
    config: UKFirmGenerationConfig,
) -> np.ndarray:
    weights = np.ones(layout.n_targets, dtype=np.float64)
    weights[layout.turnover_start : layout.turnover_start + layout.n_turnover] = (
        config.turnover_importance
    )
    weights[layout.population_start : layout.population_start + layout.n_population] = (
        config.population_importance
    )
    weights[layout.sector_start : layout.sector_start + layout.n_sectors] = (
        config.sector_importance
    )
    weights[layout.employment_start : layout.employment_start + layout.n_employment] = (
        config.employment_importance
    )
    weights[
        layout.vat_sector_start : layout.vat_sector_start + layout.n_vat_sectors
    ] = config.vat_liability_sector_importance
    weights[layout.vat_band_start : layout.vat_band_start + layout.n_vat_bands] = (
        config.vat_liability_band_importance
    )
    return weights


def _band_membership_mask(
    turnover_values: Tensor,
    band_name: str,
    threshold: float,
) -> Tensor:
    if band_name == "£1_to_Threshold":
        return (turnover_values > 0) & (turnover_values <= threshold)
    if band_name == "£Threshold_to_£150k":
        return (turnover_values > threshold) & (turnover_values <= 150)
    if band_name == "£150k_to_£300k":
        return (turnover_values > 150) & (turnover_values <= 300)
    if band_name == "£300k_to_£500k":
        return (turnover_values > 300) & (turnover_values <= 500)
    if band_name == "£500k_to_£1m":
        return (turnover_values > 500) & (turnover_values <= 1000)
    if band_name == "£1m_to_£10m":
        return (turnover_values > 1000) & (turnover_values <= 10000)
    return turnover_values > 10000


def _employment_band_index(employment: float) -> int:
    if employment <= 4:
        return 0
    if employment <= 9:
        return 1
    if employment <= 19:
        return 2
    if employment <= 49:
        return 3
    if employment <= 99:
        return 4
    if employment <= 249:
        return 5
    return 6


def _employment_targets(ons_employment: pd.DataFrame) -> dict[str, float]:
    rows = ons_employment[
        ~ons_employment["Description"].astype(str).str.contains("Total", na=False)
    ]
    return {
        band: float(rows[band].fillna(0).sum()) if band in rows else 0.0
        for band in EMPLOYMENT_BANDS
    }


def _extract_ons_total(ons_turnover: pd.DataFrame) -> int:
    sic_col = ons_turnover["SIC Code"]
    total_row = ons_turnover[sic_col.isna() | (sic_col.astype(str) == "")]
    if len(total_row) > 0 and "Total" in total_row.columns:
        return int(total_row.iloc[0]["Total"])
    sector_rows = ons_turnover[
        ~ons_turnover["Description"].astype(str).str.contains("Total", na=False)
    ]
    return int(sector_rows["Total"].sum())


def _ledger_ons_sic_band_matrix(
    facts: tuple[Mapping[str, Any] | object, ...],
    *,
    record_set_id: str,
    band_map: Mapping[str, str],
    band_dimension: str,
    table_name: str,
) -> pd.DataFrame:
    rows_by_sic: dict[str, dict[str, object]] = {}
    seen: set[tuple[str, str]] = set()
    for fact in facts:
        if not _ledger_record_matches(fact, record_set_id, "enterprise_count"):
            continue
        _validate_ledger_firm_fact(fact, record_set_id, "enterprise_count")
        sic = _ledger_sic_code(fact)
        ledger_band = _ledger_dimension_value(fact, band_dimension)
        if ledger_band not in band_map:
            continue
        key = (sic, str(ledger_band))
        if key in seen:
            raise ValueError(
                f"Duplicate Ledger UK firm fact for {record_set_id}/{sic}/"
                f"{ledger_band}."
            )
        seen.add(key)
        target_band = band_map[str(ledger_band)]
        row = rows_by_sic.setdefault(
            sic,
            {
                "SIC Code": int(sic),
                "Description": f"Ledger SIC {sic}",
            },
        )
        row[target_band] = _numeric_ledger_value(fact)

    if not rows_by_sic:
        raise ValueError(f"Ledger UK firm facts are missing {table_name} facts.")

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for sic in sorted(rows_by_sic, key=int):
        row = rows_by_sic[sic]
        for band in band_map.values():
            if band not in row:
                missing.append(f"{sic}/{band}")
                row[band] = 0.0
        row["Total"] = sum(float(row[band]) for band in band_map.values())
        rows.append(row)

    if missing:
        raise ValueError(
            f"Ledger UK firm facts are missing {table_name} cell(s): {missing}."
        )
    return pd.DataFrame(rows)


def _ledger_sic_series(
    facts: tuple[Mapping[str, Any] | object, ...],
    *,
    record_set_id: str,
    measure_id: str,
    data_vintage: str,
    value_scale: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for fact in facts:
        if not _ledger_record_matches(fact, record_set_id, measure_id):
            continue
        _validate_ledger_firm_fact(fact, record_set_id, measure_id)
        sic = _ledger_sic_code(fact)
        if sic in seen:
            raise ValueError(
                f"Duplicate Ledger UK firm fact for {record_set_id}/{sic}."
            )
        seen.add(sic)
        rows.append(
            {
                "Trade_Sector": int(sic),
                data_vintage: _numeric_ledger_value(fact) * value_scale,
            }
        )

    if not rows:
        raise ValueError(f"Ledger UK firm facts are missing {record_set_id} facts.")
    return pd.DataFrame(sorted(rows, key=lambda row: int(row["Trade_Sector"])))


def _ledger_values_by_band(
    facts: tuple[Mapping[str, Any] | object, ...],
    *,
    record_set_id: str,
    measure_id: str,
    band_map: Mapping[str, str],
) -> dict[str, float]:
    values_by_band: dict[str, float] = {}
    for fact in facts:
        if not _ledger_record_matches(fact, record_set_id, measure_id):
            continue
        _validate_ledger_firm_fact(fact, record_set_id, measure_id)
        ledger_band = _fact_at(fact, "layout", "groupby_value_id")
        if ledger_band not in band_map:
            continue
        target_band = band_map[str(ledger_band)]
        if target_band in values_by_band:
            raise ValueError(
                f"Duplicate Ledger UK firm fact for {record_set_id}/{ledger_band}."
            )
        values_by_band[target_band] = _numeric_ledger_value(fact)

    missing = [
        band_map[band] for band in band_map if band_map[band] not in values_by_band
    ]
    if missing:
        raise ValueError(
            f"Ledger UK firm facts are missing {record_set_id} band(s): {missing}."
        )
    return values_by_band


def _ledger_record_matches(
    fact: Mapping[str, Any] | object,
    record_set_id: str,
    measure_id: str,
) -> bool:
    return (
        _fact_at(fact, "layout", "record_set_id") == record_set_id
        and _fact_at(fact, "layout", "measure_id") == measure_id
    )


def _ledger_sic_code(fact: Mapping[str, Any] | object) -> str:
    sic = _ledger_dimension_value(fact, "uk.firm.sic_code") or _fact_at(
        fact, "layout", "groupby_value_id"
    )
    if sic is None:
        raise ValueError("Ledger UK firm fact is missing uk.firm.sic_code.")
    try:
        return str(int(float(str(sic)))).zfill(5)
    except ValueError as exc:
        raise ValueError(f"Ledger UK firm fact has invalid SIC code {sic!r}.") from exc


def _ledger_dimension_value(
    fact: Mapping[str, Any] | object,
    dimension: str,
) -> Any:
    value = _fact_at(fact, "dimensions", dimension)
    if value is not None:
        return value
    return _fact_at(fact, "filters", dimension)


def _validate_ledger_firm_fact(
    fact: Mapping[str, Any] | object,
    record_set_id: str,
    measure_id: str,
) -> None:
    entity_name = _fact_at(fact, "entity", "name")
    if entity_name != "firm":
        raise ValueError(
            f"Ledger UK firm fact {record_set_id}/{measure_id} has entity "
            f"{entity_name!r}; expected 'firm'."
        )
    aggregation = _fact_at(fact, "aggregation", "method")
    if aggregation != "sum":
        raise ValueError(
            f"Ledger UK firm fact {record_set_id}/{measure_id} has aggregation "
            f"{aggregation!r}; expected 'sum'."
        )


def _numeric_ledger_value(fact: Mapping[str, Any] | object) -> float:
    value = _fact_at(fact, "value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Ledger UK firm fact has non-numeric value {value!r}."
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Ledger UK firm fact has non-finite value {value!r}.")
    return numeric


def _fact_at(
    fact: Mapping[str, Any] | object,
    *path: str,
) -> Any:
    value: Any = fact
    for key in path:
        if isinstance(value, Mapping):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
        if value is None:
            return None
    return value


def _latest_hmrc_band_targets(hmrc_population_band: pd.DataFrame) -> dict[str, float]:
    return _hmrc_band_targets_for_vintage(hmrc_population_band, None)


def _latest_vat_liability_bands(hmrc_liability_band: pd.DataFrame) -> dict[str, float]:
    return _vat_liability_band_targets_for_vintage(hmrc_liability_band, None)


def _hmrc_band_targets_for_vintage(
    hmrc_population_band: pd.DataFrame,
    vintage: str | None,
) -> dict[str, float]:
    _require_columns(hmrc_population_band, HMRC_BAND_COLUMNS, "HMRC population bands")
    row = _row_for_vintage(hmrc_population_band, vintage, "HMRC population bands")
    return {column: float(row[column]) for column in HMRC_BAND_COLUMNS}


def _vat_liability_band_targets_for_vintage(
    hmrc_liability_band: pd.DataFrame,
    vintage: str | None,
) -> dict[str, float]:
    _require_columns(hmrc_liability_band, HMRC_BAND_COLUMNS, "HMRC liability bands")
    row = _row_for_vintage(hmrc_liability_band, vintage, "HMRC liability bands")
    return {column: float(row[column]) for column in HMRC_BAND_COLUMNS}


def _row_for_vintage(
    frame: pd.DataFrame,
    vintage: str | None,
    table_name: str,
) -> pd.Series:
    if len(frame) == 0:
        raise ValueError(f"{table_name} table is empty.")
    if vintage is None:
        return frame.iloc[-1]
    year_column = _vintage_row_column(frame)
    if year_column is None:
        if len(frame) == 1:
            return frame.iloc[0]
        raise ValueError(
            f"{table_name} table must include a Year or Financial_Year column "
            "when it contains multiple vintages."
        )
    matches = frame[frame[year_column].astype(str) == vintage]
    if matches.empty:
        raise ValueError(f"{table_name} table does not include vintage {vintage!r}.")
    return matches.iloc[-1]


def _vintage_row_column(frame: pd.DataFrame) -> str | None:
    for column in ("Year", "Financial_Year"):
        if column in frame.columns:
            return column
    return None


def _vintage_column_value(
    row: pd.Series,
    vintage: str,
    table_name: str,
) -> float:
    if vintage in row.index and pd.notna(row[vintage]):
        return float(row[vintage])
    id_columns = {"Trade_Sector", "SIC Code", "Description", "Year", "Financial_Year"}
    candidates = [
        column
        for column in row.index
        if column not in id_columns and pd.notna(row[column])
    ]
    if len(candidates) == 1:
        return float(row[candidates[0]])
    raise ValueError(f"{table_name} table is missing vintage column {vintage!r}.")


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    table_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} table is missing column(s): {missing}.")


def _accuracy(synthetic: float, target: float) -> float:
    if abs(target) < 1e-9:
        return 1.0 if abs(synthetic) < 1e-9 else 0.0
    return max(0.0, 1.0 - abs(synthetic - target) / abs(target))


def _as_numpy(values: np.ndarray | Tensor, dtype: type) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def _as_float32_csr(
    matrix: sparse.sparray | sparse.spmatrix | Tensor,
) -> sparse.csr_array:
    if isinstance(matrix, Tensor):
        matrix = sparse.csr_array(matrix.detach().cpu().numpy().astype(np.float32))
    return sparse.csr_array(matrix, dtype=np.float32)


def _row_lookup(codes: np.ndarray, row_codes: list[int], start: int) -> np.ndarray:
    """Target row per firm for its code among ``row_codes``; -1 when absent."""

    rows = np.full(len(codes), -1, dtype=np.int64)
    if not row_codes:
        return rows
    ordered = np.asarray(row_codes, dtype=np.int64)
    order = np.argsort(ordered, kind="stable")
    sorted_codes = ordered[order]
    position = np.searchsorted(sorted_codes, codes)
    position = np.clip(position, 0, len(sorted_codes) - 1)
    found = sorted_codes[position] == codes
    rows[found] = start + order[position[found]]
    return rows
