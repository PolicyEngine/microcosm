"""Paper parity for the experimental UK firm generator (2024-25 vintage).

The committed fixture ``tests/fixtures/uk/firm_2024_25`` holds the six
processed ONS/HMRC tables from PolicyEngine/firm-microsim-paper at aa1ce07 and
``paper_reference_2024_25.json``, golden numbers from local paper runs (see
``tools/uk_firm_paper_parity.py``). Parity is distributional, not row or weight
identity: the generators use different random streams and calibration losses.

The CI run here is thinned (window fraction 0.05, tail fraction 0.01) and
uses the generator's default 1,000 epochs, as the paper does. At 300 epochs
the net-VAT-by-band mean accuracy was 0.938, below the 0.96 bar; at 1,000 it
is 0.994. It is compared with the paper's full-census golden numbers at the
thinned tolerances of the scoping decision. Seed reproducibility uses a
separate short run pair; the main thinned solve takes about 30 seconds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_generation import (
    UKFirmGenerationConfig,
    generate_uk_firm_population,
    read_uk_firm_source_data,
)
from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs
from microcosm.build.uk_runtime.firm_universe import HMRC_BAND_NAMES
from test_support.paths import paths_for

FIXTURE_DIR = paths_for("microcosm-build").tests / "fixtures" / "uk" / "firm_2024_25"
THRESHOLD_K = 90.0
SEED = 42
CI_WINDOW_FRACTION = 0.05
CI_TAIL_FRACTION = 0.01
CI_EPOCHS = 1_000
REPRODUCIBILITY_EPOCHS = 50
PAPER_FORMULA_RATE = 0.20

HMRC_BANDS = HMRC_BAND_NAMES
CALIBRATED_VAT_BANDS = HMRC_BANDS[2:]
EMPLOYMENT_BANDS = ("0-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+")


class PaperFormulaRules:
    """Test-only rules double: the paper's 0.20 x (turnover - inputs), in GBP.

    Not UK VAT law; it lets the parity run compare draws and weights with the
    paper, whose liability model is this standard-rate approximation.
    """

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        del period
        liability = PAPER_FORMULA_RATE * (
            firms["firm_turnover"].to_numpy(dtype=np.float64)
            - firms["firm_input_costs"].to_numpy(dtype=np.float64)
        )
        return UKFirmRuleOutputs(
            firm_id=firms["firm_id"].to_numpy(dtype=np.int64),
            vat_liability_if_registered=liability,
            provenance={"rules": "paper-formula test double"},
        )


def _ci_config(seed: int = SEED, epochs: int = CI_EPOCHS) -> UKFirmGenerationConfig:
    names = {field.name for field in fields(UKFirmGenerationConfig)}
    missing = {
        "firm_rules",
        "sample_window_fraction",
        "sample_tail_fraction",
    } - names
    assert not missing, f"generator config lacks paper-parity fields: {missing}"
    # Build directly: replace() on a default (2023-24) config would keep the
    # £85k threshold its __post_init__ already resolved.
    return UKFirmGenerationConfig(
        data_vintage="2024-25",
        seed=seed,
        n_iterations=epochs,
        sample_window_fraction=CI_WINDOW_FRACTION,
        sample_tail_fraction=CI_TAIL_FRACTION,
        firm_rules=PaperFormulaRules(),
    )


@cache
def _run(seed: int = SEED) -> Any:
    data = read_uk_firm_source_data(FIXTURE_DIR)
    return generate_uk_firm_population(data, _ci_config(seed))


@cache
def _reference() -> dict[str, Any]:
    return json.loads(
        (FIXTURE_DIR / "paper_reference_2024_25.json").read_text(encoding="utf-8")
    )


@cache
def _tables() -> dict[str, pd.DataFrame]:
    return {
        "turnover": pd.read_csv(FIXTURE_DIR / "ons_firm_turnover.csv"),
        "employment": pd.read_csv(FIXTURE_DIR / "ons_firm_employment.csv"),
        "pop_band": pd.read_csv(
            FIXTURE_DIR / "hmrc_vat_population_by_turnover_band.csv"
        ),
        "pop_sector": pd.read_csv(
            FIXTURE_DIR / "hmrc_vat_population_by_sector.csv",
            dtype={"Trade_Sector": str},
        ),
        "liab_band": pd.read_csv(
            FIXTURE_DIR / "hmrc_vat_liability_by_turnover_band.csv"
        ),
    }


def _hmrc_counts() -> dict[str, float]:
    row = _tables()["pop_band"].iloc[-1]
    return {band: float(row[band]) for band in HMRC_BANDS}


def _ons_sector_rows() -> pd.DataFrame:
    turnover = _tables()["turnover"]
    sic = turnover["SIC Code"]
    rows = turnover[sic.notna() & (sic.astype(str) != "")]
    return rows[~rows["Description"].str.contains("Total", na=False)]


def _ons_band_columns() -> list[str]:
    return [
        column
        for column in _tables()["turnover"].columns
        if column not in {"SIC Code", "Description", "Total"}
    ]


def _ons_draw_rows_by_sic() -> dict[int, int]:
    columns = _ons_band_columns()
    return {
        int(row["SIC Code"]): int(row[columns].fillna(0).astype(int).sum())
        for _, row in _ons_sector_rows().iterrows()
    }


def _expected_targets() -> dict[str, float]:
    """The paper's 2024-25 target vector derived from the fixture tables."""

    tables = _tables()
    hmrc = _hmrc_counts()
    expected = {f"hmrc_vat_population/{band}": hmrc[band] for band in HMRC_BANDS}
    sectors = tables["pop_sector"][tables["pop_sector"]["Trade_Sector"] != "Total"]
    raw = {
        int(row["Trade_Sector"]): float(row.iloc[-1]) for _, row in sectors.iterrows()
    }
    scale = sum(hmrc.values()) / sum(raw.values())
    expected.update(
        {
            f"hmrc_vat_population_sector/{sic}": value * scale
            for sic, value in raw.items()
        }
    )
    employment = tables["employment"]
    employment = employment[~employment["Description"].str.contains("Total", na=False)]
    expected.update(
        {
            f"ons_firm_employment/{band}": float(employment[band].fillna(0).sum())
            for band in EMPLOYMENT_BANDS
        }
    )
    liability = tables["liab_band"].iloc[-1]
    expected.update(
        {
            f"hmrc_vat_liability/{band}": float(liability[band]) * 1000.0
            for band in CALIBRATED_VAT_BANDS
        }
    )
    return expected


def _ons_total() -> float:
    turnover = _tables()["turnover"]
    sic = turnover["SIC Code"]
    total_row = turnover[sic.isna() | (sic.astype(str) == "")]
    if len(total_row):
        return float(total_row.iloc[0]["Total"])
    return float(_ons_sector_rows()["Total"].sum())


def _band_index(turnover_k: np.ndarray) -> np.ndarray:
    edges = np.array([0.0, THRESHOLD_K, 150, 300, 500, 1000, 10000])
    return np.searchsorted(edges, turnover_k.astype(np.float64), side="left")


def _employment_index(employment: np.ndarray) -> np.ndarray:
    edges = np.array([4, 9, 19, 49, 99, 249])
    return np.searchsorted(edges, employment.astype(np.float64), side="left")


def _accuracy(estimate: float, target: float) -> float:
    return max(0.0, 1.0 - abs(estimate - target) / abs(target))


@cache
def _summary() -> dict[str, Any]:
    firms = _run().firms
    weight = firms["firm_weight"].to_numpy(dtype=np.float64)
    turnover = firms["annual_turnover_k"].to_numpy(dtype=np.float64)
    inputs = firms["annual_input_k"].to_numpy(dtype=np.float64)
    registered = firms["vat_registered"].to_numpy(dtype=bool)
    frame = firms["in_frame"].to_numpy(dtype=bool)
    band = _band_index(turnover)
    employment = _employment_index(firms["employment"].to_numpy())
    sic = firms["sic_code"].astype(int).to_numpy()
    liability_m = PAPER_FORMULA_RATE * (turnover - inputs) * weight / 1000.0
    reg_band = np.bincount(band[registered], weights=weight[registered], minlength=8)
    reg_vat = np.bincount(
        band[registered], weights=liability_m[registered], minlength=8
    )
    emp = np.bincount(employment[frame], weights=weight[frame], minlength=7)
    window = frame & (turnover > 80.0) & (turnover <= 100.0)
    return {
        "weighted_frame_population": float(weight[frame].sum()),
        "registered_traders": float(weight[registered].sum()),
        "hmrc_band_registered": dict(zip(HMRC_BANDS, reg_band.tolist(), strict=True)),
        "net_vat_by_band_m": dict(zip(HMRC_BANDS, reg_vat.tolist(), strict=True)),
        "employment_band_frame": dict(zip(EMPLOYMENT_BANDS, emp.tolist(), strict=True)),
        "sector_registered": pd.Series(weight[registered])
        .groupby(sic[registered])
        .sum()
        .to_dict(),
        "frame_mass_80_100k": float(weight[window].sum()),
    }


def _paper_full() -> dict[str, Any]:
    return _reference()["runs"]["full"]["summary"]


# --- fixture integrity ------------------------------------------------------


def test_fixture_matches_recorded_paper_provenance() -> None:
    provenance = json.loads(
        (FIXTURE_DIR / ".provenance.json").read_text(encoding="utf-8")
    )

    assert provenance["paper_commit"].startswith("aa1ce07")
    for name, entry in provenance["files"].items():
        digest = hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
    assert len(provenance["files"]) == 6
    reference_digest = hashlib.sha256(
        (FIXTURE_DIR / "paper_reference_2024_25.json").read_bytes()
    ).hexdigest()
    assert reference_digest == provenance["reference"]["sha256"]


def test_golden_full_run_reproduces_the_published_paper_headline() -> None:
    reference = json.loads(
        (FIXTURE_DIR / "paper_reference_2024_25.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (FIXTURE_DIR / ".provenance.json").read_text(encoding="utf-8")
    )
    headline = reference["paper_published_headline"]
    full = reference["runs"]["full"]["summary"]

    assert reference["paper_commit"] == provenance["paper_commit"]
    assert (
        round(full["weighted_frame_population"])
        == headline["weighted_frame_population"]
    )
    assert round(full["registered_traders"]) == headline["registered_traders"]
    assert full["pre_thinning_rows"] == headline["rows"]


def test_pre_thinning_row_count_matches_paper() -> None:
    draw_rows = sum(_ons_draw_rows_by_sic().values())
    appended = int(round(_hmrc_counts()["Negative_or_Zero"]))

    assert draw_rows + appended == 2_946_015
    assert _paper_full()["pre_thinning_rows"] == 2_946_015


# --- exact checks on the thinned run ----------------------------------------


def test_target_vector_matches_fixture_tables() -> None:
    diagnostics = _run().target_diagnostics
    expected = _expected_targets()
    actual = dict(
        zip(
            diagnostics["target_name"], diagnostics["target"].astype(float), strict=True
        )
    )
    unmatched = sorted(set(actual) - set(expected))

    # The one row outside the named families is the ONS frame population.
    assert len(unmatched) == 1, unmatched
    np.testing.assert_allclose(actual[unmatched[0]], _ons_total(), rtol=1e-6)
    assert set(expected) <= set(actual)
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=1e-6, err_msg=name)


def test_thinning_preserves_pre_thinning_totals_per_sector() -> None:
    firms = _run().firms
    frame = firms[firms["in_frame"].to_numpy(dtype=bool)]
    base_by_sic = frame.groupby(frame["sic_code"].astype(int))["base_weight"].sum()
    expected = _ons_draw_rows_by_sic()

    assert len(frame) < 0.05 * sum(expected.values())
    assert sorted(base_by_sic.index) == sorted(k for k, v in expected.items() if v)
    for sic, rows in expected.items():
        if rows:
            np.testing.assert_allclose(base_by_sic[sic], rows, rtol=1e-6, err_msg=sic)
    appended = int((~firms["in_frame"].to_numpy(dtype=bool)).sum())
    pre_thinning = round(float(frame["base_weight"].sum())) + appended
    assert pre_thinning == 2_946_015


def test_propensity_scaled_band_rows_hit_hmrc_counts_at_base_weights() -> None:
    """Row b holds p_b on band-b frame rows, so at base weights it equals HMRC_b.

    ``initial_estimate`` is the target matrix applied to the base weights, so
    this checks the propensities the generator actually used, to 1e-6.
    """

    diagnostics = _run().target_diagnostics.set_index("target_name")
    hmrc = _hmrc_counts()

    for band in HMRC_BANDS:
        row = diagnostics.loc[f"hmrc_vat_population/{band}"]
        np.testing.assert_allclose(
            row["initial_estimate"], hmrc[band], rtol=1e-6, err_msg=band
        )


def test_registration_propensities_are_hmrc_count_over_frame_mass() -> None:
    from microcosm.build.uk_runtime.firm_universe import registration_propensity

    firms = _run().firms
    frame = firms["in_frame"].to_numpy(dtype=bool)
    band = _band_index(firms["annual_turnover_k"].to_numpy())
    base = firms["base_weight"].to_numpy(dtype=np.float64)
    hmrc = _hmrc_counts()
    propensity = np.asarray(
        registration_propensity(
            band, frame, base, np.array([hmrc[b] for b in HMRC_BANDS])
        )
    )

    for index, name in enumerate(HMRC_BANDS[1:], start=1):
        mask = frame & (band == index)
        expected = min(1.0, hmrc[name] / base[mask].sum())
        np.testing.assert_allclose(propensity[mask], expected, atol=1e-6, err_msg=name)
    np.testing.assert_allclose(propensity[~frame], 1.0, atol=1e-6)


# --- distributional parity with the paper (thinned tolerances) --------------


def test_weighted_frame_population_and_registered_traders() -> None:
    summary, paper = _summary(), _paper_full()

    assert summary["weighted_frame_population"] == pytest.approx(
        paper["weighted_frame_population"], rel=0.005
    )
    assert summary["registered_traders"] == pytest.approx(
        paper["registered_traders"], rel=0.005
    )


@pytest.mark.parametrize("band", HMRC_BANDS)
def test_hmrc_band_counts_within_thinned_tolerance(band: str) -> None:
    assert _summary()["hmrc_band_registered"][band] == pytest.approx(
        _paper_full()["hmrc_band_registered"][band], rel=0.03
    )


@pytest.mark.parametrize("band", EMPLOYMENT_BANDS)
def test_employment_bands_within_tolerance(band: str) -> None:
    assert _summary()["employment_band_frame"][band] == pytest.approx(
        _paper_full()["employment_band_frame"][band], rel=0.02
    )


def test_net_vat_by_calibrated_band_mean_accuracy() -> None:
    liability = _tables()["liab_band"].iloc[-1]
    accuracies = [
        _accuracy(_summary()["net_vat_by_band_m"][band], float(liability[band]))
        for band in CALIBRATED_VAT_BANDS
    ]

    assert np.mean(accuracies) >= 0.96


def test_sector_counts_mean_accuracy() -> None:
    expected = {
        int(name.rsplit("/", 1)[1]): value
        for name, value in _expected_targets().items()
        if name.startswith("hmrc_vat_population_sector/")
    }
    synthetic = _summary()["sector_registered"]
    accuracies = [_accuracy(synthetic.get(sic, 0.0), v) for sic, v in expected.items()]

    assert np.mean(accuracies) >= 0.90


def test_frame_mass_near_threshold_within_thinned_tolerance() -> None:
    assert _summary()["frame_mass_80_100k"] == pytest.approx(
        _paper_full()["frame_mass_80_100k"], rel=0.10
    )


# --- reproducibility --------------------------------------------------------


def test_same_seed_reproduces_the_thinned_population() -> None:
    data = read_uk_firm_source_data(FIXTURE_DIR)
    config = _ci_config(epochs=REPRODUCIBILITY_EPOCHS)
    first = generate_uk_firm_population(data, config).firms
    second = generate_uk_firm_population(data, config).firms
    other_seed = generate_uk_firm_population(
        data, _ci_config(seed=SEED + 1, epochs=REPRODUCIBILITY_EPOCHS)
    ).firms

    pd.testing.assert_frame_equal(first, second)
    assert not np.array_equal(
        first["annual_turnover_k"].to_numpy(),
        other_seed["annual_turnover_k"].to_numpy(),
    )
