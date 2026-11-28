"""Tests for the UK firm BPE turnover re-pairing and open-band tail."""

from __future__ import annotations

import math
import re
from functools import cache

import numpy as np
import pandas as pd
import pytest
import torch

from microcosm.build.uk_runtime import firm_bpe_turnover as bt
from microcosm.build.uk_runtime import firm_generation as fg
from microcosm.build.uk_runtime.firm_headcount import (
    UKFirmBPESource,
    read_uk_firm_bpe_source,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURES = PATHS.tests / "fixtures" / "uk"
BPE = FIXTURES / "firm_bpe_2025"
FIRM = FIXTURES / "firm_2024_25"
MODULE = PATHS.package / "src" / "microcosm" / "build" / "uk_runtime"


@cache
def _bpe() -> UKFirmBPESource:
    return read_uk_firm_bpe_source(BPE)


@cache
def _firm_data() -> fg.UKFirmSourceData:
    return fg.read_uk_firm_source_data(FIRM)


def _synthetic(n: int = 6_000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sic = rng.choice([10, 47, 64], size=n)
    turnover = np.exp(rng.normal(5.0, 1.6, n))
    employment = np.maximum(1, np.round(np.exp(rng.normal(1.5, 1.4, n)))).astype(
        np.int64
    )
    in_frame = rng.random(n) > 0.05
    return {
        "sic": sic,
        "turnover": turnover,
        "employment": employment,
        "in_frame": in_frame,
    }


# -- constants agree with the generator ------------------------------------------


def test_band_definitions_match_generator() -> None:
    assert bt.UK_FIRM_TURNOVER_EMPLOYMENT_BANDS == fg.EMPLOYMENT_BANDS
    values = np.arange(0, 3_000)
    np.testing.assert_array_equal(
        bt.uk_firm_employment_band_index(values),
        fg.employment_band_index_array(values),
    )
    lower, upper, _ = fg.ONS_TURNOVER_BANDS[fg.ONS_OPEN_TURNOVER_BAND]
    assert bt.ONS_OPEN_TURNOVER_BAND_LABEL == fg.ONS_OPEN_TURNOVER_BAND
    assert bt.OPEN_BAND_LOWER_K == lower
    assert bt.OPEN_BAND_LOG_UNIFORM_UPPER_K == upper
    edge = bt.HMRC_TOP_BAND_LOWER_K
    assert fg.hmrc_band_name(edge, 90.0) != bt.HMRC_TOP_TURNOVER_BAND
    assert fg.hmrc_band_name(np.nextafter(edge, np.inf), 90.0) == (
        bt.HMRC_TOP_TURNOVER_BAND
    )


def test_module_avoids_banned_guards() -> None:
    text = (MODULE / "firm_bpe_turnover.py").read_text()
    for banned in ("<" + "= 9", "<" + " 10", "fewer than " + "ten"):
        assert banned not in text.lower()
    assert not re.search(r"nunique|value_counts", text)
    assert "import firm_generation" not in text
    assert "firm_generation import" not in text


# -- A. re-pairing ---------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_repair_keeps_division_multisets_and_out_of_frame(rho: float) -> None:
    d = _synthetic()
    after = bt.repair_uk_firm_employment_turnover(**d, rho=rho, seed=3)
    assert after.dtype == d["employment"].dtype
    frame = d["in_frame"]
    np.testing.assert_array_equal(after[~frame], d["employment"][~frame])
    for division in np.unique(d["sic"]):
        rows = frame & (d["sic"] == division)
        np.testing.assert_array_equal(
            np.sort(after[rows]), np.sort(d["employment"][rows])
        )


def test_repair_rho_one_is_monotone_within_division() -> None:
    d = _synthetic()
    after = bt.repair_uk_firm_employment_turnover(**d, rho=1.0, seed=3)
    for division in np.unique(d["sic"]):
        rows = np.flatnonzero(d["in_frame"] & (d["sic"] == division))
        ordered = after[rows[np.argsort(d["turnover"][rows])]]
        assert np.all(np.diff(ordered) >= 0)


def test_repair_rho_zero_ignores_turnover() -> None:
    d = _synthetic()
    base = bt.repair_uk_firm_employment_turnover(**d, rho=0.0, seed=3)
    other = dict(d, turnover=np.random.default_rng(9).random(len(d["sic"])))
    np.testing.assert_array_equal(
        bt.repair_uk_firm_employment_turnover(**other, rho=0.0, seed=3), base
    )
    # A random permutation: it moves values but not toward turnover rank.
    assert np.mean(base != d["employment"]) > 0.3
    frame = d["in_frame"]
    corr = np.corrcoef(
        np.argsort(np.argsort(d["turnover"][frame])),
        np.argsort(np.argsort(base[frame], kind="stable")),
    )[0, 1]
    assert abs(corr) < 0.05


def test_repair_is_deterministic_and_row_order_free() -> None:
    d = _synthetic()
    ids = np.arange(100, 100 + len(d["sic"]))
    first = bt.repair_uk_firm_employment_turnover(**d, rho=0.7, seed=5, firm_ids=ids)
    again = bt.repair_uk_firm_employment_turnover(**d, rho=0.7, seed=5, firm_ids=ids)
    np.testing.assert_array_equal(first, again)
    perm = np.random.default_rng(1).permutation(len(ids))
    shuffled = bt.repair_uk_firm_employment_turnover(
        **{k: v[perm] for k, v in d.items()}, rho=0.7, seed=5, firm_ids=ids[perm]
    )
    np.testing.assert_array_equal(shuffled, first[perm])
    other_seed = bt.repair_uk_firm_employment_turnover(
        **d, rho=0.7, seed=6, firm_ids=ids
    )
    assert np.any(other_seed != first)


def test_repair_rejects_bad_inputs() -> None:
    d = _synthetic(200)
    with pytest.raises(ValueError, match="rho"):
        bt.repair_uk_firm_employment_turnover(**d, rho=1.5, seed=0)
    ids = np.zeros(200, dtype=np.int64)
    with pytest.raises(ValueError, match="repeat"):
        bt.repair_uk_firm_employment_turnover(**d, rho=0.5, seed=0, firm_ids=ids)


# -- B. rho fit ------------------------------------------------------------------


def _shares(sic, turnover, employment, in_frame) -> np.ndarray:
    mask = in_frame & ~np.isin(sic, bt.SIC_SECTION_K_DIVISIONS)
    totals = np.bincount(
        bt.uk_firm_employment_band_index(employment[mask]),
        weights=turnover[mask],
        minlength=7,
    )
    return totals / totals.sum()


def test_fit_recovers_known_rho() -> None:
    d = _synthetic(40_000, seed=11)
    true_rho = 0.6
    truth = bt.repair_uk_firm_employment_turnover(**d, rho=true_rho, seed=101)
    target = _shares(d["sic"], d["turnover"], truth, d["in_frame"])
    fit = bt.fit_uk_firm_employment_turnover_rho(
        **d, base_weights=np.ones(len(d["sic"])), source=target, seed=202
    )
    assert abs(fit.rho - true_rho) < 0.05
    assert not fit.at_bound
    np.testing.assert_allclose(fit.table["target_share"], target)


def test_fit_table_is_reproduced_by_repair_and_diagnostics() -> None:
    d = _synthetic(8_000, seed=4)
    weights = np.random.default_rng(2).uniform(0.5, 2.0, len(d["sic"]))
    fit = bt.fit_uk_firm_employment_turnover_rho(
        **d, base_weights=weights, source=_bpe(), seed=7
    )
    assert 0.0 <= fit.rho <= bt.RHO_UPPER
    assert fit.sse == pytest.approx(fit.search["sse"].min())
    after = bt.repair_uk_firm_employment_turnover(**d, rho=fit.rho, seed=7)
    diag = bt.uk_firm_turnover_by_employment_band(
        d["sic"], d["turnover"], after, d["in_frame"], weights, _bpe()
    )
    np.testing.assert_allclose(diag["model_share"], fit.table["model_share_fitted"])
    np.testing.assert_allclose(diag["bpe_share"], fit.table["target_share"])
    before = bt.uk_firm_turnover_by_employment_band(
        d["sic"], d["turnover"], d["employment"], d["in_frame"], weights, _bpe()
    )
    np.testing.assert_allclose(before["model_share"], fit.table["model_share_before"])


def test_fit_flags_bound() -> None:
    d = _synthetic(4_000, seed=8)
    # Two headcounts, all target turnover in the top band: the error falls
    # as the pairing tightens, so the search ends at its upper bound.
    d["employment"] = np.where(np.arange(4_000) % 2 == 0, 1, 300)
    target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    fit = bt.fit_uk_firm_employment_turnover_rho(
        **d, base_weights=np.ones(len(d["sic"])), source=target, seed=2
    )
    assert fit.rho == bt.RHO_UPPER
    assert fit.at_bound


def test_fit_accepts_synthetic_bpe_source() -> None:
    labels = [
        x
        for band in bt.UK_FIRM_TURNOVER_EMPLOYMENT_BANDS
        for x in bt.UK_FIRM_BPE_BANDS_BY_ONS_BAND[band]
    ]
    rows = [("Table 2", "whole_economy", label, 10.0) for label in labels]
    rows += [
        ("Table 2", "whole_economy", "All", 10.0 * len(labels)),
        ("Table 1", "private_sector", "With no employees (unregistered)", 5.0),
    ]
    size = pd.DataFrame(
        rows, columns=["table", "sector", "size_band", "turnover_million"]
    )
    source = UKFirmBPESource(legal_status=pd.DataFrame(), size_bands=size, sha256={})
    table = bt.uk_firm_bpe_turnover_by_band(source)
    assert table["bpe_turnover_million"].iloc[0] == 15.0
    d = _synthetic(2_000)
    fit = bt.fit_uk_firm_employment_turnover_rho(
        **d, base_weights=np.ones(2_000), source=source, seed=1
    )
    np.testing.assert_allclose(fit.table["target_share"], table["bpe_share"])


# -- C. open-band tail -----------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 5_000])
def test_tail_draw_bounds_and_exact_mean(n: int) -> None:
    alpha, cap, lower = 0.83, 6.2e6, 5_000.0
    uniforms = np.random.default_rng(n).random(n)
    values = bt.draw_uk_firm_open_band_turnover(uniforms, alpha, cap, lower)
    assert np.all((values >= lower) & (values <= cap))
    assert values.mean() == pytest.approx(
        bt._truncated_pareto_mean(alpha, lower, cap), rel=1e-9
    )
    over = np.mean(values > bt.HMRC_TOP_BAND_LOWER_K)
    implied = bt.truncated_pareto_exceedance(
        bt.HMRC_TOP_BAND_LOWER_K, alpha, lower, cap
    )
    assert abs(over - implied) <= 1.0 / n + 1e-12


def test_tail_draw_uses_only_the_uniforms_ranks() -> None:
    uniforms = np.random.default_rng(3).random(1_000)
    state = np.random.get_state()[1].copy()
    torch_state = torch.get_rng_state().clone()
    values = bt.draw_uk_firm_open_band_turnover(uniforms, 0.9, 1e6)
    np.testing.assert_array_equal(np.random.get_state()[1], state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(
        bt.draw_uk_firm_open_band_turnover(uniforms**3, 0.9, 1e6), values
    )
    assert np.all(np.diff(values[np.argsort(uniforms)]) > 0)


def test_tail_draw_strata_have_exact_means() -> None:
    uniforms = np.random.default_rng(0).random(301)
    strata = np.arange(301) % 3 == 0
    values = bt.draw_uk_firm_open_band_turnover(uniforms, 0.8, 1e6, strata=strata)
    mean = bt._truncated_pareto_mean(0.8, 5_000.0, 1e6)
    for flag in (True, False):
        assert values[strata == flag].mean() == pytest.approx(mean, rel=1e-9)


def test_open_band_uniforms_invert_log_uniform_draw() -> None:
    unit = np.linspace(0.0, 0.999, 50)
    drawn = 5_000.0 * 10.0**unit
    np.testing.assert_allclose(bt.uk_firm_open_band_uniforms(drawn), unit, atol=1e-12)


def _fixture_tail_inputs():
    data = _firm_data()
    rng = np.random.default_rng(0)
    sic = np.array([10, 47, 64, 10, 47, 64, 10, 10])
    turnover = np.array([20.0, 300.0, 4_000.0, 6_000.0, 9_000.0, 7_000.0, 0.0, 1e3])
    in_frame = np.array([True] * 6 + [False, True])
    open_band = turnover >= 5_000.0
    weights = rng.uniform(1_000.0, 5_000.0, sic.size)
    return data, sic, turnover, in_frame, open_band, weights


def test_tail_parameters_follow_fixture_arithmetic() -> None:
    data, sic, turnover, in_frame, open_band, weights = _fixture_tail_inputs()
    hmrc = fg._hmrc_band_targets_for_vintage(data.hmrc_population_band, "2024-25")
    tail = bt.uk_firm_open_band_tail_parameters(
        data.ons_turnover,
        hmrc,
        _bpe(),
        sic=sic,
        turnover_k=turnover,
        in_frame=in_frame,
        open_band=open_band,
        weights=weights,
    )
    ons = pd.read_csv(FIRM / "ons_firm_turnover.csv")
    ons_open = float(ons.loc[ons["SIC Code"].notna(), "5000+"].sum())
    assert ons_open == float(ons.loc[ons["Description"] == "Total", "5000+"].iloc[0])
    hmrc_csv = pd.read_csv(FIRM / "hmrc_vat_population_by_turnover_band.csv")
    hmrc_top = float(hmrc_csv["Greater_than_£10m"].iloc[0])
    p = hmrc_top / ons_open
    assert tail.p_over_10m == pytest.approx(p)
    assert tail.alpha == pytest.approx(-math.log(p) / math.log(2.0))
    size = pd.read_csv(BPE / "bpe_size_bands.csv")
    whole = size.query(
        "table == 'Table 2' and sector == 'whole_economy' and size_band == 'All'"
    )["turnover_million"].iloc[0]
    unreg = size.query(
        "table == 'Table 1' and size_band == 'With no employees (unregistered)'"
    )["turnover_million"].iloc[0]
    non_k = in_frame & ~np.isin(sic, [64, 65, 66])
    closed = (turnover * weights)[non_k & ~open_band].sum()
    firms = weights[non_k & open_band].sum()
    target = ((whole - unreg) * 1_000.0 - closed) / firms
    assert tail.target_mean_k == pytest.approx(target)
    assert bt._truncated_pareto_mean(
        tail.alpha, tail.lower_k, tail.cap_k
    ) == pytest.approx(target, rel=1e-10)
    assert tail.model_p_over_10m == pytest.approx(
        bt.truncated_pareto_exceedance(10_000.0, tail.alpha, 5_000.0, tail.cap_k)
    )
    assert tail.model_p_over_10m < tail.p_over_10m


def test_tail_parameters_reject_unreachable_mean() -> None:
    data, sic, turnover, in_frame, open_band, weights = _fixture_tail_inputs()
    hmrc = {"Greater_than_£10m": 1.0}  # p tiny -> alpha above one
    with pytest.raises(ValueError, match="beyond any cap"):
        bt.uk_firm_open_band_tail_parameters(
            data.ons_turnover,
            hmrc,
            _bpe(),
            sic=sic,
            turnover_k=turnover,
            in_frame=in_frame,
            open_band=open_band,
            weights=weights,
        )


# -- D. pins and BPE target ------------------------------------------------------


def test_bpe_band_target_and_pins_equal_fixture_arithmetic() -> None:
    size = pd.read_csv(BPE / "bpe_size_bands.csv")
    whole = size[(size["table"] == "Table 2") & (size["sector"] == "whole_economy")]
    turnover = whole.set_index("size_band")["turnover_million"]
    unreg = float(
        size.loc[
            size["size_band"] == "With no employees (unregistered)", "turnover_million"
        ].iloc[0]
    )
    expected = [
        turnover["Zero or 1 employee"] + turnover["2 to 4"] - unreg,
        turnover["5 to 9"],
        turnover["10 to 19"],
        turnover["20 to 49"],
        turnover["50 to 99"],
        turnover["100 to 199"] + turnover["200 to 249"],
        turnover["250 to 499"] + turnover["500 to 999"] + turnover["1000 or more"],
    ]
    table = bt.uk_firm_bpe_turnover_by_band(_bpe())
    np.testing.assert_allclose(table["bpe_turnover_million"], expected)
    # The bands cover the whole-economy total up to BPE rounding.
    assert sum(expected) + unreg == pytest.approx(turnover["All"], abs=5.0)
    data = _firm_data()
    pins = bt.uk_firm_bpe_turnover_pins(_bpe(), data.ons_turnover, data.hmrc_bands)
    assert pins["bpe_whole_economy_turnover"]["value"] == turnover["All"]
    assert pins["bpe_unregistered_turnover"]["value"] == unreg
    for band, value in zip(bt.UK_FIRM_TURNOVER_EMPLOYMENT_BANDS, expected, strict=True):
        assert pins[f"bpe_registered_turnover_{band}"]["value"] == value
    ons = pd.read_csv(FIRM / "ons_firm_turnover.csv")
    hmrc_csv = pd.read_csv(FIRM / "hmrc_vat_population_by_turnover_band.csv")
    assert pins["ons_open_band_enterprises"]["value"] == float(
        ons.loc[ons["SIC Code"].notna(), "5000+"].sum()
    )
    assert pins["hmrc_over_10m_traders"]["value"] == float(
        hmrc_csv["Greater_than_£10m"].iloc[-1]
    )
    assert all(pin["source"] for pin in pins.values())


def test_status_and_method_are_labelled() -> None:
    assert bt.UK_FIRM_BPE_TURNOVER_STATUS == "assumed"
    assert "Business population estimates 2025" in bt.UK_FIRM_BPE_TURNOVER_METHOD
    assert "BPE_2025_detailed_tables.xlsx" in bt.__doc__
