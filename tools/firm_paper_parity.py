#!/usr/bin/env python3
"""Compare Microcosm's experimental UK firm generator with the firm-microsim paper.

The paper (PolicyEngine/firm-microsim-paper) generates a synthetic UK firm
population from processed ONS and HMRC tables. Microcosm ports that generator
into ``microcosm.build.uk_runtime.firm_generation``. Parity is distributional,
not row or weight identity: the two generators use different random streams
and different calibration losses (the paper minimises symmetric relative error
with dropout and an L1 penalty; Microcosm minimises capped MAPE).

Usage::

    # Compare Microcosm (full census draw) with the committed paper reference.
    uv run --no-sync python tools/firm_paper_parity.py --full

    # Thinned Microcosm run (window 0.30, tails 0.05, the paper's fast preset).
    uv run --no-sync python tools/firm_paper_parity.py --fast

    # Also re-run the paper generator from a checkout and compare with that.
    uv run --no-sync python tools/firm_paper_parity.py --full \
        --paper-root ../firm-microsim-paper

``--paper-root`` runs the paper inside its own environment
(``<paper-root>/.venv/bin/python`` when present, else ``uv run --project``)
by invoking this file's ``paper-extract`` subcommand there. The same
subcommand produced the committed ``paper_reference_2024_25.json`` fixture.

VAT rules: Microcosm keeps VAT law out of the generator and requires a rules
object. ``--rules paper-formula`` supplies a PARITY-ONLY double defined in this
file that reproduces the paper's standard-rate approximation,
0.20 x (turnover - inputs). It is not a statement of UK VAT law and is not
shipped in ``src``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "tests"
    / "fixtures"
    / "uk"
    / "firm_2024_25"
)
REFERENCE_FILE = "paper_reference_2024_25.json"
VINTAGE = "2024-25"
THRESHOLD_K = 90.0
SEED = 42
PAPER_VAT_RATE = 0.20

HMRC_BANDS: tuple[str, ...] = (
    "Negative_or_Zero",
    "£1_to_Threshold",
    "£Threshold_to_£150k",
    "£150k_to_£300k",
    "£300k_to_£500k",
    "£500k_to_£1m",
    "£1m_to_£10m",
    "Greater_than_£10m",
)
# The paper calibrates net VAT on every positive band except £1_to_Threshold.
CALIBRATED_VAT_BANDS: tuple[str, ...] = HMRC_BANDS[2:]
EMPLOYMENT_BANDS: tuple[str, ...] = (
    "0-4",
    "5-9",
    "10-19",
    "20-49",
    "50-99",
    "100-249",
    "250+",
)
EMPLOYMENT_UPPER: tuple[float, ...] = (4, 9, 19, 49, 99, 249, float("inf"))
HMRC_UPPER: tuple[float, ...] = (0.0, THRESHOLD_K, 150, 300, 500, 1000, 10000)

# Thinning presets: (window fraction, tail fraction).
FAST_PRESET = (0.30, 0.05)
CI_PRESET = (0.05, 0.01)


# ---------------------------------------------------------------------------
# Shared summary (pure pandas: runs in either environment)
# ---------------------------------------------------------------------------
def hmrc_band_index(turnover_k: np.ndarray, threshold_k: float) -> np.ndarray:
    """Map turnover (GBP thousands) to HMRC band indices 0-7."""

    edges = np.array([0.0, threshold_k, 150, 300, 500, 1000, 10000])
    return np.searchsorted(edges, np.asarray(turnover_k, dtype=np.float64), "left")


def employment_band_index(employment: np.ndarray) -> np.ndarray:
    """Map employment counts to ONS employment band indices 0-6."""

    edges = np.array(EMPLOYMENT_UPPER[:-1])
    return np.searchsorted(edges, np.asarray(employment, dtype=np.float64), "left")


def summarise_firms(
    firms: pd.DataFrame,
    *,
    weight_column: str,
    threshold_k: float = THRESHOLD_K,
) -> dict[str, Any]:
    """Return the parity statistics for one generated firm table.

    Net VAT is recomputed with the paper formula on both sides so the
    comparison isolates weights and draws from the rules double.
    """

    weight = firms[weight_column].to_numpy(dtype=np.float64)
    turnover = firms["annual_turnover_k"].to_numpy(dtype=np.float64)
    inputs = firms["annual_input_k"].to_numpy(dtype=np.float64)
    registered = firms["vat_registered"].to_numpy(dtype=bool)
    if "in_frame" in firms.columns:
        frame = firms["in_frame"].to_numpy(dtype=bool)
    else:
        frame = turnover > 0
    band = hmrc_band_index(turnover, threshold_k)
    emp_band = employment_band_index(firms["employment"].to_numpy())
    sic = firms["sic_code"].astype(int).to_numpy()
    liability_m = PAPER_VAT_RATE * (turnover - inputs) * weight / 1000.0

    def by_band(mask: np.ndarray, values: np.ndarray) -> dict[str, float]:
        sums = np.bincount(band[mask], weights=values[mask], minlength=8)
        return {name: float(sums[i]) for i, name in enumerate(HMRC_BANDS)}

    emp_sums = np.bincount(emp_band[frame], weights=weight[frame], minlength=7)
    sector = pd.Series(weight[registered]).groupby(sic[registered]).sum().sort_index()
    window = frame & (turnover > 80.0) & (turnover <= 100.0)
    summary: dict[str, Any] = {
        "rows_total": int(len(firms)),
        "rows_in_frame": int(frame.sum()),
        "rows_appended": int((~frame).sum()),
        "weighted_frame_population": float(weight[frame].sum()),
        "weighted_all_rows": float(weight.sum()),
        "registered_traders": float(weight[registered].sum()),
        "hmrc_band_registered": by_band(registered, weight),
        "hmrc_band_frame": by_band(frame, weight),
        "employment_band_frame": {
            name: float(emp_sums[i]) for i, name in enumerate(EMPLOYMENT_BANDS)
        },
        "net_vat_by_band_m": by_band(registered, liability_m),
        "sector_registered": {str(int(k)): float(v) for k, v in sector.items()},
        "frame_mass_80_100k": float(weight[window].sum()),
    }
    if "base_weight" in firms.columns:
        base = firms["base_weight"].to_numpy(dtype=np.float64)
        summary["base_weighted_frame_rows"] = float(base[frame].sum())
        summary["base_weighted_frame_by_band"] = by_band(frame, base)
        summary["base_weighted_frame_by_sic"] = {
            str(int(k)): float(v)
            for k, v in pd.Series(base[frame]).groupby(sic[frame]).sum().items()
        }
    return summary


def accuracy(estimate: float, target: float) -> float:
    """Paper accuracy: ``max(0, 1 - |estimate - target| / |target|)``."""

    if abs(target) < 1e-9:
        return 1.0 if abs(estimate) < 1e-9 else 0.0
    return max(0.0, 1.0 - abs(estimate - target) / abs(target))


# ---------------------------------------------------------------------------
# Official targets read straight from the processed CSVs
# ---------------------------------------------------------------------------
def official_targets(processed_dir: Path) -> dict[str, Any]:
    """Return the target vector the paper derives from the six processed CSVs."""

    turnover = pd.read_csv(processed_dir / "ons_firm_turnover.csv")
    employment = pd.read_csv(processed_dir / "ons_firm_employment.csv")
    pop_band = pd.read_csv(processed_dir / "hmrc_vat_population_by_turnover_band.csv")
    pop_sector = pd.read_csv(
        processed_dir / "hmrc_vat_population_by_sector.csv", dtype={"Trade_Sector": str}
    )
    liab_band = pd.read_csv(processed_dir / "hmrc_vat_liability_by_turnover_band.csv")

    sic_col = turnover["SIC Code"]
    total_row = turnover[sic_col.isna() | (sic_col.astype(str) == "")]
    if len(total_row) and "Total" in total_row.columns:
        ons_total = int(total_row.iloc[0]["Total"])
    else:
        rows = turnover[~turnover["Description"].str.contains("Total", na=False)]
        ons_total = int(rows["Total"].sum())
    sector_rows = turnover[sic_col.notna() & (sic_col.astype(str) != "")]
    sector_rows = sector_rows[
        ~sector_rows["Description"].str.contains("Total", na=False)
    ]
    band_cols = [
        c for c in turnover.columns if c not in {"SIC Code", "Description", "Total"}
    ]
    draw_rows = int(sector_rows[band_cols].fillna(0).astype(int).sum().sum())
    draw_rows_by_sic = {
        str(int(r["SIC Code"])): int(r[band_cols].fillna(0).astype(int).sum())
        for _, r in sector_rows.iterrows()
    }

    hmrc_bands = {b: float(pop_band.iloc[-1][b]) for b in HMRC_BANDS}
    vat_bands = {b: float(liab_band.iloc[-1][b]) for b in HMRC_BANDS[1:]}
    emp_rows = employment[~employment["Description"].str.contains("Total", na=False)]
    emp_targets = {b: float(emp_rows[b].fillna(0).sum()) for b in EMPLOYMENT_BANDS}
    sector = pop_sector[pop_sector["Trade_Sector"] != "Total"]
    band_total = float(sum(hmrc_bands.values()))
    sector_total = float(sector.iloc[:, -1].astype(float).sum())
    scale = band_total / sector_total if sector_total > 0 else 1.0
    sector_targets = {
        str(int(r["Trade_Sector"])): float(r.iloc[-1]) * scale
        for _, r in sector.iterrows()
    }
    return {
        "ons_total": ons_total,
        "ons_draw_rows": draw_rows,
        "ons_draw_rows_by_sic": draw_rows_by_sic,
        "pre_thinning_rows": draw_rows + int(round(hmrc_bands["Negative_or_Zero"])),
        "hmrc_bands": hmrc_bands,
        "vat_liability_bands_m": vat_bands,
        "employment_bands": emp_targets,
        "sector_registered_scaled": sector_targets,
        "sector_scale": scale,
    }


def propensities_from_summary(
    summary: Mapping[str, Any], targets: Mapping[str, Any]
) -> dict[str, float]:
    """Registration propensity per positive band: HMRC count / base frame mass."""

    mass = summary.get("base_weighted_frame_by_band")
    if mass is None:
        return {}
    out: dict[str, float] = {}
    for name in HMRC_BANDS[1:]:
        if mass[name] > 0:
            out[name] = min(1.0, targets["hmrc_bands"][name] / mass[name])
    return out


def accuracies(
    summary: Mapping[str, Any], targets: Mapping[str, Any]
) -> dict[str, float]:
    """Paper-style accuracy per calibrated dimension against official targets."""

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "hmrc_bands": mean(
            [
                accuracy(summary["hmrc_band_registered"][b], targets["hmrc_bands"][b])
                for b in HMRC_BANDS
            ]
        ),
        "ons_population": accuracy(
            summary["weighted_frame_population"], targets["ons_total"]
        ),
        "employment": mean(
            [
                accuracy(
                    summary["employment_band_frame"][b], targets["employment_bands"][b]
                )
                for b in EMPLOYMENT_BANDS
            ]
        ),
        "sector": mean(
            [
                accuracy(summary["sector_registered"].get(k, 0.0), v)
                for k, v in targets["sector_registered_scaled"].items()
            ]
        ),
        "vat_liability_band": mean(
            [
                accuracy(
                    summary["net_vat_by_band_m"][b], targets["vat_liability_bands_m"][b]
                )
                for b in CALIBRATED_VAT_BANDS
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Paper side (runs inside the paper's environment)
# ---------------------------------------------------------------------------
def _paper_extract(mode: str, out_path: Path, n_iterations: int | None) -> None:
    import importlib

    import firm_microsim
    from firm_microsim.config import Config

    # ``firm_microsim.generate`` the attribute is the function; load the module.
    paper_calibration = importlib.import_module("firm_microsim.calibration")
    paper_generate = importlib.import_module("firm_microsim.generate")

    captured: dict[str, Any] = {}

    original_propensity = paper_calibration.registration_propensity

    def capture_propensity(band_indices, frame_mask, base_weights, hmrc_bands):
        prop = original_propensity(band_indices, frame_mask, base_weights, hmrc_bands)
        bands = band_indices.cpu().numpy()
        frame = frame_mask.cpu().numpy()
        values = prop.cpu().numpy()
        base = base_weights.cpu().numpy().astype(np.float64)
        captured["propensities"] = {
            name: float(values[frame & (bands == i)][0])
            for i, name in enumerate(HMRC_BANDS)
            if i > 0 and (frame & (bands == i)).any()
        }
        captured["base_weighted_frame_by_band"] = {
            name: float(base[frame & (bands == i)].sum())
            for i, name in enumerate(HMRC_BANDS)
        }
        return prop

    paper_calibration.registration_propensity = capture_propensity

    original_base = paper_generate.generate_base_firms

    def capture_base(*args, **kwargs):
        sic, turnover = original_base(*args, **kwargs)
        captured["pre_thinning_frame_rows"] = int(len(sic))
        return sic, turnover

    paper_generate.generate_base_firms = capture_base

    overrides: dict[str, Any] = {"seed": SEED}
    if mode == "fast":
        overrides.update(
            sample_window_fraction=FAST_PRESET[0], sample_tail_fraction=FAST_PRESET[1]
        )
    elif mode == "ci":
        overrides.update(
            sample_window_fraction=CI_PRESET[0], sample_tail_fraction=CI_PRESET[1]
        )
    if n_iterations is not None:
        overrides["n_iterations"] = n_iterations
    config = Config.for_vintage(VINTAGE, **overrides)

    started = time.perf_counter()
    firms, report = paper_generate.generate(config, write=False, return_report=True)
    runtime = time.perf_counter() - started

    processed = Path(config.processed_dir)
    targets = official_targets(processed)
    summary = summarise_firms(firms, weight_column="weight", threshold_k=THRESHOLD_K)
    summary["pre_thinning_frame_rows"] = captured.get("pre_thinning_frame_rows")
    summary["pre_thinning_rows"] = (
        captured["pre_thinning_frame_rows"] + summary["rows_appended"]
    )
    summary["propensities"] = captured.get("propensities", {})
    summary["base_weighted_frame_by_band"] = captured.get("base_weighted_frame_by_band")
    summary["accuracy_paper_report"] = {
        "hmrc_bands": report.hmrc_bands,
        "ons_population": report.ons_population,
        "employment": report.employment,
        "sector": report.sector,
        "vat_liability_band": report.vat_liability_band,
        "overall": report.overall,
        "vat_liability_sector_diagnostic": report.vat_liability_sector,
        "vat_liability_below_threshold_diagnostic": (
            report.vat_liability_below_threshold
        ),
    }
    summary["accuracy_recomputed"] = accuracies(summary, targets)
    payload = {
        "metadata": {
            "paper_package_version": getattr(firm_microsim, "__version__", None),
            "paper_commit": _git_head(processed),
            "vintage": VINTAGE,
            "vat_threshold_k": THRESHOLD_K,
            "seed": SEED,
            "mode": mode,
            "sample_window_fraction": config.sample_window_fraction,
            "sample_tail_fraction": config.sample_tail_fraction,
            "n_iterations": config.n_iterations,
            "runtime_seconds": round(runtime, 1),
            "command": (
                "python tools/firm_paper_parity.py paper-extract "
                f"--mode {mode} (paper generate(write=False))"
            ),
        },
        "targets": targets,
        "summary": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_paper(paper_root: Path, mode: str, out_path: Path) -> dict[str, Any]:
    """Run the paper generator in its own environment and return its summary."""

    python = paper_root / ".venv" / "bin" / "python"
    if python.exists():
        command = [str(python)]
    else:
        command = ["uv", "run", "--project", str(paper_root), "python"]
    command += [str(Path(__file__).resolve()), "paper-extract", "--mode", mode]
    command += ["--out", str(out_path)]
    subprocess.run(command, check=True, cwd=paper_root)
    return json.loads(out_path.read_text())


# ---------------------------------------------------------------------------
# Microcosm side
# ---------------------------------------------------------------------------
class PaperFormulaVATRules:
    """PARITY-ONLY VAT rules double: 0.20 x (turnover - inputs) for every firm.

    Reproduces the paper's standard-rate approximation so parity runs compare
    weights and draws, not VAT law. Implements both the legacy
    ``net_liability_k`` evaluator and the ``UKFirmRules.evaluate`` protocol.
    """

    provenance = {"rules": "paper-formula", "status": "parity-only double"}

    def net_liability_k(
        self,
        *,
        turnover_k: np.ndarray,
        input_cost_k: np.ndarray,
        vat_registered: np.ndarray,
        data_vintage: str,
    ) -> np.ndarray:
        del vat_registered, data_vintage
        return PAPER_VAT_RATE * (
            np.asarray(turnover_k, dtype=np.float64)
            - np.asarray(input_cost_k, dtype=np.float64)
        )

    def evaluate(self, firms: pd.DataFrame, *, period: int):
        from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs

        del period
        liability = PAPER_VAT_RATE * (
            firms["firm_turnover"].to_numpy(dtype=np.float64)
            - firms["firm_input_costs"].to_numpy(dtype=np.float64)
        )
        return UKFirmRuleOutputs(
            firm_id=firms["firm_id"].to_numpy(dtype=np.int64),
            vat_liability_if_registered=liability,
            provenance=dict(self.provenance),
        )


def microcosm_config(
    *,
    window_fraction: float,
    tail_fraction: float,
    n_iterations: int | None = None,
    seed: int = SEED,
):
    """Build a Microcosm generator config wired to the paper-formula double."""

    from microcosm.build.uk_runtime.firm_generation import UKFirmGenerationConfig

    names = {f.name for f in fields(UKFirmGenerationConfig)}
    rules = PaperFormulaVATRules()
    values: dict[str, Any] = {"data_vintage": VINTAGE, "seed": seed}
    if "firm_rules" in names:
        values["firm_rules"] = rules
    else:
        values["vat_rule_evaluator"] = rules
    if window_fraction < 1.0 or tail_fraction < 1.0:
        if "sample_window_fraction" not in names:
            raise SystemExit(
                "UKFirmGenerationConfig has no sample_window_fraction; "
                "thinned runs need the stratified-thinning generator."
            )
        values["sample_window_fraction"] = window_fraction
        values["sample_tail_fraction"] = tail_fraction
    if n_iterations is not None:
        values["n_iterations"] = n_iterations
    # Construct directly so __post_init__ resolves the 2024-25 £90k threshold;
    # replace() on a default config would keep the 2023-24 £85k it resolved.
    return UKFirmGenerationConfig(**values)


def run_microcosm(processed_dir: Path, config) -> tuple[dict[str, Any], float, Any]:
    """Run the Microcosm generator and return (summary, runtime seconds, result)."""

    from microcosm.build.uk_runtime.firm_generation import (
        generate_uk_firm_population,
        read_uk_firm_source_data,
    )

    data = read_uk_firm_source_data(processed_dir)
    started = time.perf_counter()
    result = generate_uk_firm_population(data, config)
    runtime = time.perf_counter() - started
    summary = summarise_firms(result.firms, weight_column="firm_weight")
    return summary, runtime, result


# ---------------------------------------------------------------------------
# Parity table
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    name: str
    paper: float | None
    microcosm: float | None
    kind: str  # "exact", "rel", "abs", "min", "info"
    tolerance: float

    @property
    def diff(self) -> float | None:
        if self.paper is None or self.microcosm is None:
            return None
        if self.kind == "rel":
            return (self.microcosm - self.paper) / self.paper if self.paper else 0.0
        return self.microcosm - self.paper

    @property
    def passed(self) -> bool | None:
        if self.microcosm is None or self.kind == "info":
            return None
        if self.kind == "min":
            return self.microcosm >= self.tolerance
        diff = self.diff
        if diff is None:
            return None
        return abs(diff) <= self.tolerance


def parity_checks(
    paper: Mapping[str, Any],
    micro: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    thinned: bool,
) -> list[Check]:
    """Build the provisional parity checks from the scoping decision."""

    band_tol = 0.03 if thinned else 0.01
    window_tol = 0.10 if thinned else 0.05
    checks: list[Check] = []
    pre_rows = micro.get("base_weighted_frame_rows")
    checks.append(
        Check(
            "pre-thinning row count",
            paper.get("pre_thinning_rows"),
            None if pre_rows is None else round(pre_rows) + micro["rows_appended"],
            "exact",
            0.0,
        )
    )
    # Propensities depend on each generator's own turnover draw, so they are
    # shown side by side for information only; the 1e-6 identity (HMRC count /
    # own base-weighted frame mass) is asserted on Microcosm's run in the test.
    micro_prop = propensities_from_summary(micro, targets)
    for band, value in paper.get("propensities", {}).items():
        checks.append(
            Check(f"propensity {band} (info)", value, micro_prop.get(band), "info", 0.0)
        )
    checks.append(
        Check(
            "weighted frame population",
            paper["weighted_frame_population"],
            micro["weighted_frame_population"],
            "rel",
            0.005,
        )
    )
    checks.append(
        Check(
            "registered traders",
            paper["registered_traders"],
            micro["registered_traders"],
            "rel",
            0.005,
        )
    )
    for band in HMRC_BANDS:
        checks.append(
            Check(
                f"HMRC registered {band}",
                paper["hmrc_band_registered"][band],
                micro["hmrc_band_registered"][band],
                "rel",
                band_tol,
            )
        )
    for band in EMPLOYMENT_BANDS:
        checks.append(
            Check(
                f"employment {band}",
                paper["employment_band_frame"][band],
                micro["employment_band_frame"][band],
                "rel",
                0.02,
            )
        )
    micro_acc = accuracies(micro, targets)
    paper_acc = paper.get("accuracy_recomputed", accuracies(paper, targets))
    checks.append(
        Check(
            "net VAT by calibrated band (mean accuracy)",
            paper_acc["vat_liability_band"],
            micro_acc["vat_liability_band"],
            "min",
            0.96,
        )
    )
    checks.append(
        Check(
            "sector counts (mean accuracy)",
            paper_acc["sector"],
            micro_acc["sector"],
            "min",
            0.90,
        )
    )
    checks.append(
        Check(
            "frame mass £80-100k",
            paper["frame_mass_80_100k"],
            micro["frame_mass_80_100k"],
            "rel",
            window_tol,
        )
    )
    return checks


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind in {"min"} or (abs(value) < 10 and kind != "exact"):
        return f"{value:.6g}"
    return f"{value:,.0f}"


def render_table(checks: list[Check]) -> str:
    """Return a markdown parity table."""

    lines = [
        "| Check | Paper | Microcosm | Diff | Tolerance | Result |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for check in checks:
        diff = check.diff
        if check.kind == "rel" and diff is not None:
            diff_text = f"{diff:+.2%}"
            tol_text = f"±{check.tolerance:.1%}"
        elif check.kind == "min":
            diff_text = "" if diff is None else f"{diff:+.4f}"
            tol_text = f"≥ {check.tolerance:.0%}"
        elif check.kind == "info":
            diff_text = "n/a" if diff is None else f"{diff:+.4f}"
            tol_text = "info"
        elif check.kind == "exact":
            diff_text = "n/a" if diff is None else f"{diff:+,.0f}"
            tol_text = "exact"
        else:
            diff_text = "n/a" if diff is None else f"{diff:+.2e}"
            tol_text = f"±{check.tolerance:g}"
        passed = check.passed
        result = "n/a" if passed is None else ("pass" if passed else "FAIL")
        lines.append(
            f"| {check.name} | {_fmt(check.paper, check.kind)} | "
            f"{_fmt(check.microcosm, check.kind)} | {diff_text} | {tol_text} | {result} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "paper-extract":
        parser = argparse.ArgumentParser(prog="firm_paper_parity.py paper-extract")
        parser.add_argument("--mode", choices=["full", "fast", "ci"], default="full")
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--n-iterations", type=int, default=None)
        args = parser.parse_args(argv[1:])
        _paper_extract(args.mode, args.out, args.n_iterations)
        return 0

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", dest="mode", action="store_const", const="full")
    mode.add_argument("--fast", dest="mode", action="store_const", const="fast")
    mode.add_argument("--ci", dest="mode", action="store_const", const="ci")
    parser.set_defaults(mode="full")
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=None,
        help="Paper checkout; when given, re-runs the paper generator (full).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=FIXTURE_DIR,
        help="Processed 2024-25 CSV directory (default: committed fixture).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=FIXTURE_DIR / REFERENCE_FILE,
        help="Paper reference JSON used when --paper-root is not given.",
    )
    parser.add_argument(
        "--rules",
        choices=["paper-formula"],
        default="paper-formula",
        help="VAT rules double (parity only).",
    )
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.paper_root is not None:
        out = Path(os.environ.get("TMPDIR", "/tmp")) / "firm_paper_reference.json"
        paper_payload = run_paper(args.paper_root.resolve(), "full", out)
    else:
        paper_payload = json.loads(args.reference.read_text())["runs"]["full"]

    window, tail = {"full": (1.0, 1.0), "fast": FAST_PRESET, "ci": CI_PRESET}[args.mode]
    config = microcosm_config(
        window_fraction=window, tail_fraction=tail, n_iterations=args.n_iterations
    )
    micro, runtime, _ = run_microcosm(args.processed_dir, config)
    targets = official_targets(args.processed_dir)
    checks = parity_checks(
        paper_payload["summary"], micro, targets, thinned=args.mode != "full"
    )
    print(
        f"Microcosm mode={args.mode} (window {window}, tails {tail}), "
        f"seed {SEED}, rules {args.rules} (parity-only), "
        f"{micro['rows_total']:,} rows, {runtime:.1f} s"
    )
    meta = paper_payload["metadata"]
    print(
        f"Paper reference: mode={meta['mode']}, commit {meta.get('paper_commit')}, "
        f"{meta['runtime_seconds']} s"
    )
    print()
    print(render_table(checks))
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps({"microcosm": micro, "runtime_seconds": runtime}, indent=2)
        )
    return 0 if all(c.passed is not False for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
