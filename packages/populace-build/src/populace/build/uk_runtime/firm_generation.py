"""Experimental UK firm-population generator migrated from firm-microsim-paper.

This module ports the paper repository's ONS/HMRC firm generator into
``populace.build.uk_runtime`` without carrying the paper repo's processed CSVs.
Callers should pass Ledger consumer facts through
``uk_firm_source_data_from_ledger_facts``. The explicit source-table reader is
kept only for migration comparisons against the paper repository.

Scope is intentionally narrow and labelled experimental. Populace production
calibration targets must be materialized from Ledger target profiles; the
processed ONS/HMRC tables accepted here are a migration harness only, not a
production target source. Ledger-backed firm target activation and Axiom
VAT-rule execution are later integration steps, so this module does not claim
production firm microsimulation support.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import Tensor

from populace.calibrate.solve import (
    _optimize as _calibrate_optimize,
)
from populace.calibrate.solve import (
    _torch_constraint_matrix as _calibrate_torch_constraint_matrix,
)
from populace.calibrate.solve import (
    default_target_loss_scales,
    relative_error_loss,
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
LEDGER_ONS_TURNOVER_RECORD_SET = (
    "ons.uk_business.cy2025.enterprise_count.by_turnover_band"
)
LEDGER_ONS_EMPLOYMENT_RECORD_SET = (
    "ons.uk_business.cy2025.enterprise_count.by_employment_band"
)
LEDGER_ONS_SIC_TURNOVER_RECORD_SET = (
    "ons.uk_business.cy2025.enterprise_count.by_sic_turnover_band"
)
LEDGER_ONS_SIC_EMPLOYMENT_RECORD_SET = (
    "ons.uk_business.cy2025.enterprise_count.by_sic_employment_band"
)
LEDGER_HMRC_POPULATION_RECORD_SET = (
    "hmrc.vat.fy2024_25.registered_trader_count.by_turnover_band"
)
LEDGER_HMRC_LIABILITY_RECORD_SET = (
    "hmrc.vat.fy2024_25.net_liability.by_turnover_band"
)
LEDGER_HMRC_POPULATION_SIC_RECORD_SET = (
    "hmrc.vat.fy2024_25.registered_trader_count.by_sic"
)
LEDGER_HMRC_LIABILITY_SIC_RECORD_SET = (
    "hmrc.vat.fy2024_25.net_liability.by_sic"
)
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

HMRC_BAND_COLUMNS: tuple[str, ...] = (
    "Negative_or_Zero",
    "£1_to_Threshold",
    "£Threshold_to_£150k",
    "£150k_to_£300k",
    "£300k_to_£500k",
    "£500k_to_£1m",
    "£1m_to_£10m",
    "Greater_than_£10m",
)
VAT_LIABILITY_BANDS: tuple[str, ...] = HMRC_BAND_COLUMNS[1:]
EMPLOYMENT_BANDS: tuple[str, ...] = (
    "0-4",
    "5-9",
    "10-19",
    "20-49",
    "50-99",
    "100-249",
    "250+",
)

ONS_TURNOVER_BANDS: dict[str, tuple[float, float, float]] = {
    "0-49": (0.0, 49.0, 24.5),
    "50-99": (50.0, 99.0, 74.5),
    "100-249": (100.0, 249.0, 174.5),
    "250-499": (250.0, 499.0, 374.5),
    "500-999": (500.0, 999.0, 749.5),
    "1000-4999": (1000.0, 4999.0, 2999.5),
    "5000+": (5000.0, 50000.0, 15000.0),
}
ONS_EMPLOYMENT_BANDS: dict[str, tuple[float, float, float]] = {
    "0-4": (1.0, 4.0, 2.5),
    "5-9": (5.0, 9.0, 7.0),
    "10-19": (10.0, 19.0, 14.5),
    "20-49": (20.0, 49.0, 34.5),
    "50-99": (50.0, 99.0, 74.5),
    "100-249": (100.0, 249.0, 174.5),
    "250+": (250.0, 2000.0, 400.0),
}

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


@dataclass(frozen=True)
class UKFirmGenerationConfig:
    """Configuration for the experimental UK firm generator."""

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
    sector_importance: float = 1.0
    employment_importance: float = 1.0
    vat_liability_sector_importance: float = 1.0
    vat_liability_band_importance: float = 2.0
    calibrate_vat_liability_sector: bool = False
    input_files: dict[str, str] = field(default_factory=lambda: dict(INPUT_FILES))

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
    """Target row layout in the generated calibration matrix."""

    n_sectors: int
    n_vat_sectors: int

    @property
    def n_turnover(self) -> int:
        return 7

    @property
    def n_employment(self) -> int:
        return len(EMPLOYMENT_BANDS)

    @property
    def n_vat_bands(self) -> int:
        return len(VAT_LIABILITY_BANDS)

    @property
    def turnover_start(self) -> int:
        return 0

    @property
    def sector_start(self) -> int:
        return self.turnover_start + self.n_turnover

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
    """Core Populace calibration result for pre-zero-turnover base firms."""

    weights: Tensor
    loss_trajectory: np.ndarray
    initial_loss: float
    final_loss: float
    target_loss_weights: np.ndarray
    target_loss_scales: np.ndarray


@dataclass(frozen=True)
class UKFirmGenerationResult:
    """Generated firm rows plus validation and calibration diagnostics."""

    firms: pd.DataFrame
    validation: UKFirmValidationReport
    calibration: UKFirmCalibrationResult
    target_diagnostics: pd.DataFrame
    config: UKFirmGenerationConfig


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


def uk_firm_source_data_from_ledger_facts(
    facts: Iterable[Mapping[str, Any] | object],
    *,
    data_vintage: str = "2024-25",
) -> UKFirmSourceData:
    """Build UK firm inputs from Ledger consumer facts.

    The accepted rows are the Ledger/Arch consumer-fact shape emitted from the
    ``uk_firms`` target profile. Ledger values are the calibration target source;
    this adapter only reshapes them into the generator's temporary matrix
    contract while firm support is being migrated.
    """

    fact_rows = tuple(facts)
    ons_turnover = _ledger_ons_sic_band_matrix(
        fact_rows,
        record_set_id=LEDGER_ONS_SIC_TURNOVER_RECORD_SET,
        band_map=LEDGER_ONS_TURNOVER_BANDS,
        band_dimension="uk.firm.turnover_band",
        table_name="ONS SIC turnover",
    )
    ons_employment = _ledger_ons_sic_band_matrix(
        fact_rows,
        record_set_id=LEDGER_ONS_SIC_EMPLOYMENT_RECORD_SET,
        band_map=LEDGER_ONS_EMPLOYMENT_BANDS,
        band_dimension="uk.firm.employment_band",
        table_name="ONS SIC employment",
    )
    hmrc_population_values = _ledger_values_by_band(
        fact_rows,
        record_set_id=LEDGER_HMRC_POPULATION_RECORD_SET,
        measure_id="vat_registered_trader_count",
        band_map=LEDGER_HMRC_BANDS,
    )
    hmrc_liability_values_gbp = _ledger_values_by_band(
        fact_rows,
        record_set_id=LEDGER_HMRC_LIABILITY_RECORD_SET,
        measure_id="net_vat_liability",
        band_map=LEDGER_HMRC_BANDS,
    )
    hmrc_liability_values_m = {
        band: value / 1_000_000.0
        for band, value in hmrc_liability_values_gbp.items()
    }
    hmrc_population_band = pd.DataFrame(
        [{"Financial_Year": data_vintage, **hmrc_population_values}]
    )
    hmrc_population_sector = _ledger_sic_series(
        fact_rows,
        record_set_id=LEDGER_HMRC_POPULATION_SIC_RECORD_SET,
        measure_id="vat_registered_trader_count",
        data_vintage=data_vintage,
        value_scale=1.0,
    )
    hmrc_liability_band = pd.DataFrame(
        [{"Financial_Year": data_vintage, **hmrc_liability_values_m}]
    )
    hmrc_liability_sector = _ledger_sic_series(
        fact_rows,
        record_set_id=LEDGER_HMRC_LIABILITY_SIC_RECORD_SET,
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
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    hmrc_bands = _hmrc_band_targets_for_vintage(
        data.hmrc_population_band,
        cfg.data_vintage,
    )
    base_sic, base_turnover = generate_base_firms(data.ons_turnover, cfg.device)
    if len(base_sic) == 0:
        raise ValueError("UK firm generation requires at least one ONS firm row.")
    base_input = generate_input_values(base_turnover, base_sic, cfg.device)
    base_employment = assign_employment(
        len(base_sic),
        data.ons_employment,
        cfg.device,
    )
    base_vat_registered = assign_vat_flags(base_turnover, hmrc_bands, cfg)
    employment_band_indices = torch.tensor(
        [_employment_band_index(value.item()) for value in base_employment],
        dtype=torch.long,
        device=cfg.device,
    )

    target_matrix, target_values, layout, target_names = build_firm_target_matrix(
        cfg,
        base_turnover,
        base_sic,
        base_input,
        employment_band_indices,
        base_vat_registered,
        data,
    )
    calibration = solve_firm_weights(cfg, target_matrix, target_values, layout)
    weights = calibration.weights

    (
        final_sic,
        final_turnover,
        final_input,
        final_employment,
        final_weights,
        final_vat_registered,
    ) = _add_zero_turnover_firms(
        base_sic,
        base_turnover,
        base_input,
        base_employment,
        weights,
        base_vat_registered,
        hmrc_bands,
        data.ons_employment,
        cfg.device,
    )
    firms = _assemble_firm_rows(
        final_sic,
        final_turnover,
        final_input,
        final_employment,
        final_weights,
        final_vat_registered,
        cfg,
    )
    validation = validate_uk_firm_population(firms, data, cfg)
    diagnostics = target_diagnostics(
        target_matrix,
        target_values,
        weights,
        target_names,
    )
    return UKFirmGenerationResult(
        firms=firms,
        validation=validation,
        calibration=calibration,
        target_diagnostics=diagnostics,
        config=cfg,
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
            turnovers.append(_draw_band_turnover(count, lower, upper, device).cpu().numpy())

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
    """Draw annual input expenditure in thousands of pounds."""

    n_firms = len(turnover_values)
    if n_firms == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    ratios = torch.distributions.Beta(4.0, 2.0).sample((n_firms,)).to(device)
    scaled = 0.3 + ratios * 1.0
    sector_noise = torch.randn(n_firms, device=device) * 0.15
    sic_np = sic_codes.detach().cpu().numpy()
    neg_mask = np.isin(sic_np, list(NEGATIVE_LIABILITY_SECTORS))
    high_mask = np.isin(sic_np, list(HIGH_LIABILITY_SECTORS))
    neg_t = torch.tensor(neg_mask, device=device)
    high_t = torch.tensor(high_mask, device=device)
    scaled = scaled + neg_t.float() * (torch.rand(n_firms, device=device) * 0.3)
    scaled = scaled - high_t.float() * (torch.rand(n_firms, device=device) * 0.2)
    final_ratios = torch.clamp(scaled + sector_noise, 0.6, 1.5)
    return torch.where(
        turnover_values > 0,
        turnover_values * final_ratios,
        torch.zeros_like(turnover_values),
    )


def assign_employment(
    n_firms: int,
    ons_employment: pd.DataFrame,
    device: str = "cpu",
) -> Tensor:
    """Assign firm employment counts from ONS employment-band shares."""

    if n_firms == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    sector_rows = ons_employment[
        ~ons_employment["Description"].astype(str).str.contains("Total", na=False)
    ]
    band_counts = {
        band: int(sector_rows[band].fillna(0).sum()) if band in sector_rows else 0
        for band in EMPLOYMENT_BANDS
    }
    total = sum(band_counts.values()) or 1
    values: list[float] = []
    for band in EMPLOYMENT_BANDS:
        target = int(round(n_firms * band_counts[band] / total))
        if target <= 0:
            continue
        lower, upper, midpoint = ONS_EMPLOYMENT_BANDS[band]
        if band == "0-4":
            draws = torch.randint(1, 5, (target,), device=device).float()
        elif band == "250+":
            log_mean = torch.log(torch.tensor(float(midpoint), device=device))
            draws = torch.normal(log_mean, 0.8, (target,), device=device).exp()
            draws = torch.clamp(draws, lower, upper).round()
        else:
            u = torch.rand(target, device=device)
            beta_approx = u.pow(0.5) * (1 - u).pow(2.0)
            draws = (lower + beta_approx * (upper - lower)).round()
        values.extend(draws.cpu().numpy())

    np.random.shuffle(values)
    if len(values) < n_firms:
        values.extend([1] * (n_firms - len(values)))
    else:
        values = values[:n_firms]
    return torch.tensor(values, dtype=torch.float32, device=device)


def assign_vat_flags(
    turnover_values: Tensor,
    hmrc_bands: dict[str, float],
    config: UKFirmGenerationConfig,
) -> Tensor:
    """Assign VAT-registration flags."""

    threshold = float(config.vat_threshold)
    device = turnover_values.device
    below = (turnover_values > 0) & (turnover_values <= threshold)
    flags = turnover_values > threshold
    below_indices = below.nonzero(as_tuple=False).flatten()
    if len(below_indices) == 0:
        return flags
    target_below = int(round(max(0.0, float(hmrc_bands["£1_to_Threshold"]))))
    n_select = min(len(below_indices), target_below)
    if n_select > 0:
        selected = below_indices[
            torch.randperm(len(below_indices), device=device)[:n_select]
        ]
        flags[selected] = True
    return flags


def build_firm_target_matrix(
    config: UKFirmGenerationConfig,
    turnover_values: Tensor,
    sic_codes: Tensor,
    input_values: Tensor,
    employment_band_indices: Tensor,
    vat_registered: Tensor,
    data: UKFirmSourceData,
) -> tuple[Tensor, Tensor, UKFirmTargetLayout, tuple[str, ...]]:
    """Construct the firm calibration target matrix and target vector."""

    threshold = float(config.vat_threshold)
    hmrc_bands = _hmrc_band_targets_for_vintage(
        data.hmrc_population_band,
        config.data_vintage,
    )
    vat_liability_bands = _vat_liability_band_targets_for_vintage(
        data.hmrc_liability_band,
        config.data_vintage,
    )
    n_firms = len(turnover_values)
    sector_rows = data.hmrc_population_sector[
        data.hmrc_population_sector["Trade_Sector"] != "Total"
    ].copy()
    if config.calibrate_vat_liability_sector:
        vat_sector_rows = data.hmrc_liability_sector[
            data.hmrc_liability_sector["Trade_Sector"] != "Total"
        ].copy()
    else:
        vat_sector_rows = data.hmrc_liability_sector.iloc[0:0].copy()

    layout = UKFirmTargetLayout(
        n_sectors=len(sector_rows),
        n_vat_sectors=len(vat_sector_rows),
    )
    matrix = torch.zeros(layout.n_targets, n_firms, device=config.device)
    band_indices = map_to_hmrc_band_indices(turnover_values, threshold)

    target_names: list[str] = []
    for row, band_idx in enumerate(range(1, 8)):
        matrix[row, (band_indices == band_idx) & vat_registered] = 1.0
        target_names.append(f"hmrc_vat_population/{HMRC_BAND_COLUMNS[band_idx]}")

    for offset, (_, sector_row) in enumerate(sector_rows.iterrows()):
        row = layout.sector_start + offset
        sic_code = int(sector_row["Trade_Sector"])
        matrix[row, (sic_codes == sic_code) & vat_registered] = 1.0
        target_names.append(f"hmrc_vat_population_sector/{sic_code}")

    for band_idx, band in enumerate(EMPLOYMENT_BANDS):
        row = layout.employment_start + band_idx
        matrix[row, employment_band_indices == band_idx] = 1.0
        target_names.append(f"ons_firm_employment/{band}")

    vat_liability = turnover_values - input_values
    for offset, (_, vat_row) in enumerate(vat_sector_rows.iterrows()):
        row = layout.vat_sector_start + offset
        sic_code = int(vat_row["Trade_Sector"])
        mask = (sic_codes == sic_code) & vat_registered
        matrix[row, mask] = vat_liability[mask]
        target_names.append(f"hmrc_vat_liability_sector/{sic_code}")

    for offset, band in enumerate(VAT_LIABILITY_BANDS):
        row = layout.vat_band_start + offset
        mask = _band_membership_mask(turnover_values, band, threshold) & vat_registered
        matrix[row, mask] = vat_liability[mask]
        target_names.append(f"hmrc_vat_liability/{band}")

    turnover_targets = [
        hmrc_bands[band]
        for band in HMRC_BAND_COLUMNS
        if band != "Negative_or_Zero"
    ]
    sector_targets = [
        _vintage_column_value(row, config.data_vintage, "HMRC population sectors")
        for _, row in sector_rows.iterrows()
    ]
    ons_emp_rows = data.ons_employment[
        ~data.ons_employment["Description"].astype(str).str.contains("Total", na=False)
    ]
    employment_targets = [
        float(ons_emp_rows[band].fillna(0).sum()) if band in ons_emp_rows else 0.0
        for band in EMPLOYMENT_BANDS
    ]
    vat_sector_targets = [
        _vintage_column_value(row, config.data_vintage, "HMRC liability sectors")
        * 1000.0
        for _, row in vat_sector_rows.iterrows()
    ]
    vat_band_targets = [
        float(vat_liability_bands[band]) * 1000.0 for band in VAT_LIABILITY_BANDS
    ]
    targets = torch.tensor(
        turnover_targets
        + sector_targets
        + employment_targets
        + vat_sector_targets
        + vat_band_targets,
        dtype=torch.float32,
        device=config.device,
    )
    return matrix, targets, layout, tuple(target_names)


def solve_firm_weights(
    config: UKFirmGenerationConfig,
    target_matrix: Tensor,
    target_values: Tensor,
    layout: UKFirmTargetLayout,
) -> UKFirmCalibrationResult:
    """Solve firm weights using Populace's shared calibration optimizer."""

    _, n_firms = target_matrix.shape
    initial_weights = np.ones(n_firms, dtype=np.float64)
    matrix_np = target_matrix.detach().cpu().numpy().astype(np.float64, copy=False)
    matrix = sparse.csr_array(matrix_np)
    target_np = target_values.detach().cpu().numpy().astype(np.float64, copy=False)
    loss_weights = _target_loss_weights(layout, config)
    loss_scales = default_target_loss_scales(target_np)

    torch.manual_seed(config.seed)
    weights, trajectory = _calibrate_optimize(
        _calibrate_torch_constraint_matrix(matrix),
        torch.tensor(target_np, dtype=torch.float32),
        torch.tensor(loss_weights, dtype=torch.float32),
        torch.tensor(loss_scales, dtype=torch.float32),
        config.target_loss_cap,
        initial_weights,
        epochs=config.n_iterations,
        learning_rate=config.learning_rate,
        conserve_mass=config.conserve_mass,
        max_weight_ratio=config.max_weight_ratio,
        l0_lambda=0.0,
        l2_lambda=0.0,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
    )
    initial_estimates = matrix @ initial_weights
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
    )


def optimize_firm_weights(
    config: UKFirmGenerationConfig,
    target_matrix: Tensor,
    target_values: Tensor,
    layout: UKFirmTargetLayout,
) -> Tensor:
    """Return core-solved firm weights for callers that only need the vector."""

    return solve_firm_weights(config, target_matrix, target_values, layout).weights


def target_diagnostics(
    target_matrix: Tensor,
    target_values: Tensor,
    weights: Tensor,
    names: tuple[str, ...],
) -> pd.DataFrame:
    """Return per-target diagnostics for the optimized weight vector."""

    estimates = (target_matrix @ weights).detach().cpu().numpy()
    targets = target_values.detach().cpu().numpy()
    relative_error = np.divide(
        estimates - targets,
        np.where(np.abs(targets) < 1e-9, 1.0, targets),
    )
    initial = np.ones_like(weights.detach().cpu().numpy(), dtype=np.float64)
    initial_estimates = (target_matrix.detach().cpu().numpy() @ initial).astype(
        np.float64,
        copy=False,
    )
    return pd.DataFrame(
        {
            "target_name": names,
            "initial_estimate": initial_estimates,
            "estimate": estimates,
            "target": targets,
            "relative_error": relative_error,
        }
    )


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
    weighted_bands = registered.groupby("hmrc_band")["firm_weight"].sum()
    hmrc_accs = []
    for band_name, target in hmrc_bands.items():
        hmrc_accs.append(_accuracy(float(weighted_bands.get(band_name, 0.0)), target))

    employment_targets = _employment_targets(data.ons_employment)
    df["employment_band"] = df["employment"].apply(employment_band_name)
    weighted_employment = df.groupby("employment_band")["firm_weight"].sum()
    employment_accs = [
        _accuracy(float(weighted_employment.get(band, 0.0)), target)
        for band, target in employment_targets.items()
    ]

    sector_rows = data.hmrc_population_sector[
        data.hmrc_population_sector["Trade_Sector"] != "Total"
    ]
    weighted_sector = registered.groupby("sic_numeric")["firm_weight"].sum()
    sector_accs = [
        _accuracy(
            float(weighted_sector.get(int(row["Trade_Sector"]), 0.0)),
            _vintage_column_value(row, config.data_vintage, "HMRC population sectors"),
        )
        for _, row in sector_rows.iterrows()
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

    weighted_liability_band = registered.groupby("hmrc_band")["weighted_liability_m"].sum()
    liability_band_accs = [
        _accuracy(
            float(weighted_liability_band.get(band, 0.0)),
            # HMRC liability targets are in millions; firm rows store thousands.
            float(vat_liability_bands[band]),
        )
        for band in VAT_LIABILITY_BANDS
    ]

    return UKFirmValidationReport(
        hmrc_bands=float(np.mean(hmrc_accs)) if hmrc_accs else 0.0,
        ons_population=_accuracy(float(df["firm_weight"].sum()), float(data.ons_total)),
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
) -> Tensor:
    band_width = upper - lower
    noise_std = max(25.0, band_width * 0.2)
    base = lower + torch.rand(count, device=device) * band_width
    noise = torch.normal(0, noise_std, (count,), device=device)
    return torch.clamp(base + noise, min=0.1)


def _add_zero_turnover_firms(
    sic_codes: Tensor,
    turnover: Tensor,
    input_values: Tensor,
    employment: Tensor,
    weights: Tensor,
    vat_registered: Tensor,
    hmrc_bands: dict[str, float],
    ons_employment: pd.DataFrame,
    device: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    target = int(round(hmrc_bands["Negative_or_Zero"]))
    if target <= 0 or len(sic_codes) == 0:
        return sic_codes, turnover, input_values, employment, weights, vat_registered
    add_sics = _allocate_zero_turnover_sics(sic_codes, target)
    if not add_sics:
        return sic_codes, turnover, input_values, employment, weights, vat_registered
    n_add = len(add_sics)
    zero_employment = assign_employment(n_add, ons_employment, device)
    return (
        torch.cat([sic_codes, torch.tensor(add_sics, dtype=torch.int64, device=device)]),
        torch.cat([turnover, torch.zeros(n_add, dtype=torch.float32, device=device)]),
        torch.cat([input_values, torch.zeros(n_add, dtype=torch.float32, device=device)]),
        torch.cat([employment, zero_employment]),
        torch.cat([weights, torch.ones(n_add, dtype=torch.float32, device=device)]),
        torch.cat(
            [
                vat_registered,
                torch.ones(n_add, dtype=torch.bool, device=device),
            ]
        ),
    )


def _allocate_zero_turnover_sics(sic_codes: Tensor, target: int) -> list[int]:
    unique_sics, counts = torch.unique(sic_codes, return_counts=True)
    total = len(sic_codes)
    exact_allocations = target * counts.float() / total
    allocations = torch.floor(exact_allocations).to(torch.int64)
    remainder = target - int(allocations.sum().item())
    if remainder > 0:
        fractional_order = torch.argsort(
            exact_allocations - allocations.float(),
            descending=True,
        )
        allocations[fractional_order[:remainder]] += 1
    add_sics: list[int] = []
    for sic, n_add in zip(unique_sics, allocations, strict=True):
        add_sics.extend([int(sic.item())] * int(n_add.item()))
    return add_sics


def _assemble_firm_rows(
    sic_codes: Tensor,
    turnover: Tensor,
    input_values: Tensor,
    employment: Tensor,
    weights: Tensor,
    vat_registered: Tensor,
    config: UKFirmGenerationConfig,
) -> pd.DataFrame:
    sic_np = sic_codes.detach().cpu().numpy().astype(int)
    turnover_np = turnover.detach().cpu().numpy()
    input_np = input_values.detach().cpu().numpy()
    firm_ids = np.arange(1, len(sic_np) + 1, dtype=np.int64)
    return pd.DataFrame(
        {
            "firm_id": firm_ids,
            "source_firm_id": firm_ids,
            "source_firm_key": [
                f"{config.data_vintage}:experimental_uk_firm:{firm_id}"
                for firm_id in firm_ids
            ],
            "data_vintage": config.data_vintage,
            "sic_code": [str(sic).zfill(5) for sic in sic_np],
            "annual_turnover_k": turnover_np,
            "annual_input_k": input_np,
            "vat_liability_k": turnover_np - input_np,
            "employment": employment.detach().cpu().numpy().astype(int),
            "firm_weight": weights.detach().cpu().numpy(),
            "vat_registered": vat_registered.detach().cpu().numpy().astype(bool),
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
            raise ValueError(f"Duplicate Ledger UK firm fact for {record_set_id}/{sic}.")
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

    missing = [band_map[band] for band in band_map if band_map[band] not in values_by_band]
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
    sic = (
        _ledger_dimension_value(fact, "uk.firm.sic_code")
        or _fact_at(fact, "layout", "groupby_value_id")
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
        raise ValueError(f"Ledger UK firm fact has non-numeric value {value!r}.") from exc
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
