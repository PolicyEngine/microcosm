"""The opt-in large-firm headcount tail against the default 2024-25 draw.

``UKFirmGenerationConfig.large_employment_mean`` replaces the default 250+
headcount draw with a truncated Pareto tail fitted to a published mean. That
mean describes businesses in the ONS frame, so with the option on the
out-of-frame stratum (the HMRC zero-turnover traders) draws below 250 and
the 250+ band holds frame rows only. The default build is pinned by
``test_uk_firm_paper_parity.py``; these tests show the option changes
headcounts only, and only where the default draws 250 or more.
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_generation import (
    LARGE_EMPLOYMENT_CAP,
    LARGE_EMPLOYMENT_MEAN,
    UKFirmGenerationConfig,
    generate_uk_firm_population,
    read_uk_firm_source_data,
)
from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs
from test_support.paths import paths_for

FIXTURE_DIR = paths_for("microcosm-build").tests / "fixtures" / "uk" / "firm_2024_25"
EPOCHS = 20


class _StandardRateDouble:
    """Test-only rules double: 0.20 x (turnover - inputs), in pounds."""

    def evaluate(self, firms: pd.DataFrame, *, period: object) -> UKFirmRuleOutputs:
        del period
        liability = 0.20 * (
            firms["firm_turnover"].to_numpy(dtype=np.float64)
            - firms["firm_input_costs"].to_numpy(dtype=np.float64)
        )
        return UKFirmRuleOutputs(
            firm_id=firms["firm_id"].to_numpy(dtype=np.int64),
            vat_liability_if_registered=liability,
            provenance={"rules": "standard-rate test double"},
        )


def _config() -> UKFirmGenerationConfig:
    return UKFirmGenerationConfig(
        data_vintage="2024-25",
        seed=42,
        n_iterations=EPOCHS,
        sample_window_fraction=0.05,
        sample_tail_fraction=0.01,
        firm_rules=_StandardRateDouble(),
    )


@cache
def _firms(large_employment_mean: float | None) -> pd.DataFrame:
    data = read_uk_firm_source_data(FIXTURE_DIR)
    config = dataclasses.replace(_config(), large_employment_mean=large_employment_mean)
    return generate_uk_firm_population(data, config).firms


def _ons_large_firm_count() -> float:
    table = read_uk_firm_source_data(FIXTURE_DIR).ons_employment
    sectors = table[~table["Description"].astype(str).str.contains("Total", na=False)]
    return float(sectors["250+"].fillna(0).sum())


def test_default_config_leaves_the_large_employment_mean_off() -> None:
    assert UKFirmGenerationConfig().large_employment_mean is None
    assert _config().large_employment_mean is None
    default = _firms(None)
    # The default keeps the paper's clamp, and its out-of-frame rows reach
    # the open band: the behaviour the option removes.
    assert default["employment"].max() <= 2_000
    outside = default.loc[~default["in_frame"], "employment"]
    assert (outside >= 250).any()


def test_the_option_changes_headcounts_only_where_the_default_draws_large() -> None:
    default = _firms(None)
    large = _firms(LARGE_EMPLOYMENT_MEAN)

    # Weights, turnover, registration and every other column are the
    # default's: employment target rows count frame rows only, and the frame
    # rows keep their bands.
    pd.testing.assert_frame_equal(
        large.drop(columns="employment"), default.drop(columns="employment")
    )
    legacy = default["employment"].to_numpy()
    employment = large["employment"].to_numpy()
    np.testing.assert_array_equal(employment[legacy < 250], legacy[legacy < 250])
    in_frame = large["in_frame"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(employment >= 250, (legacy >= 250) & in_frame)


def test_out_of_frame_rows_draw_below_the_large_band() -> None:
    large = _firms(LARGE_EMPLOYMENT_MEAN)
    outside = large.loc[~large["in_frame"], "employment"].to_numpy()

    assert len(outside) > 0
    assert outside.min() >= 1
    assert outside.max() <= 249


def test_weighted_large_firm_count_and_mean() -> None:
    default = _firms(None)
    large = _firms(LARGE_EMPLOYMENT_MEAN)
    employment = large["employment"].to_numpy(dtype=np.float64)
    weights = large["firm_weight"].to_numpy(dtype=np.float64)
    base_weights = large["base_weight"].to_numpy(dtype=np.float64)
    in_frame = large["in_frame"].to_numpy(dtype=bool)
    band = employment >= 250

    # The weighted 250+ count is the frame's alone: the default's in-frame
    # 250+ weight, which the solve holds near the ONS 250+ enterprise count.
    legacy_band = default["employment"].to_numpy() >= 250
    legacy_weights = default["firm_weight"].to_numpy(dtype=np.float64)
    count = float(weights[band].sum())
    assert count == pytest.approx(float(legacy_weights[legacy_band & in_frame].sum()))
    assert count < float(legacy_weights[legacy_band].sum())
    assert count == pytest.approx(_ons_large_firm_count(), rel=0.10)

    # The tail is fitted at base weights over the frame's 250+ rows, so
    # their base-weighted mean is the configured mean; calibration moves the
    # weighted mean a little.
    base_mean = float(
        (base_weights * employment)[band].sum() / base_weights[band].sum()
    )
    assert base_mean == pytest.approx(LARGE_EMPLOYMENT_MEAN, rel=1e-3)
    mean = float((weights * employment)[band].sum() / weights[band].sum())
    assert mean == pytest.approx(LARGE_EMPLOYMENT_MEAN, rel=0.10)
    assert 2_000 < employment.max() <= LARGE_EMPLOYMENT_CAP
