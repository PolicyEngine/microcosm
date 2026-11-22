"""End to end: generated UK firms, ONS attribute draws, policyengine-uk table.

A thinned 2024-25 build on the paper-parity fixture gets region and legal
status from the ONS UK Business 2025 attribute tables and is exported with
policyengine-uk's firm column names. The ONS Table 15 grid (region x legal
status), which the draws never read, is the held-out check.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_attributes import (
    assign_uk_firm_attributes,
    read_uk_firm_attribute_source_data,
    uk_firm_attribute_fallback_counts,
)
from microcosm.build.uk_runtime.firm_generation import (
    UKFirmGenerationConfig,
    UKFirmGenerationResult,
    generate_uk_firm_population,
    read_uk_firm_source_data,
)
from microcosm.build.uk_runtime.firm_paybill import (
    assign_uk_firm_paybill,
    read_uk_firm_paybill_source,
    uk_firm_paybill_fallback_counts,
)
from microcosm.build.uk_runtime.firm_rules import UKFirmRuleOutputs
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_EXTENDED_TABLE_COLUMNS,
    UK_FIRM_PAYBILL_TABLE_COLUMNS,
    UK_FIRM_TABLE_COLUMNS,
    to_policyengine_uk_firm_table,
    validate_uk_firm_table,
)
from test_support.paths import paths_for

FIXTURES = paths_for("microcosm-build").tests / "fixtures" / "uk"
SEED = 42


class _StandardRateDouble:
    """Test-only rules double, 0.20 x (turnover - inputs); not UK VAT law."""

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
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


@cache
def _generated() -> UKFirmGenerationResult:
    config = UKFirmGenerationConfig(
        data_vintage="2024-25",
        seed=SEED,
        n_iterations=20,
        sample_window_fraction=0.02,
        sample_tail_fraction=0.005,
        firm_rules=_StandardRateDouble(),
    )
    return generate_uk_firm_population(
        read_uk_firm_source_data(FIXTURES / "firm_2024_25"), config
    )


@cache
def _with_attributes() -> pd.DataFrame:
    source = read_uk_firm_attribute_source_data(FIXTURES / "firm_attributes_2025")
    return assign_uk_firm_attributes(_generated().firms, source, seed=SEED)


def test_attribute_draws_leave_generated_columns_unchanged() -> None:
    firms = _generated().firms
    attributed = _with_attributes()

    pd.testing.assert_frame_equal(attributed[list(firms.columns)], firms)


def test_extended_export_validates_and_keeps_the_base_columns() -> None:
    result = _generated()
    threshold_k = result.config.vat_threshold
    base = to_policyengine_uk_firm_table(result.firms, threshold_k=threshold_k)
    extended = to_policyengine_uk_firm_table(
        _with_attributes(), threshold_k=threshold_k
    )

    validate_uk_firm_table(extended)
    assert tuple(extended.columns) == UK_FIRM_EXTENDED_TABLE_COLUMNS
    assert tuple(base.columns) == UK_FIRM_TABLE_COLUMNS
    pd.testing.assert_frame_equal(extended[list(UK_FIRM_TABLE_COLUMNS)], base)
    assert not extended.isna().to_numpy().any()


def test_in_frame_firms_use_cells_and_out_of_frame_firms_their_row() -> None:
    source = read_uk_firm_attribute_source_data(FIXTURES / "firm_attributes_2025")
    firms = _with_attributes()
    counts = uk_firm_attribute_fallback_counts(firms, source)
    by_level = counts.set_index(["attribute", "level"])["firms"]
    in_frame = int(firms["in_frame"].to_numpy(dtype=bool).sum())
    out_of_frame = len(firms) - in_frame

    assert in_frame > 0
    assert out_of_frame > 0
    for attribute in ("ons_region_code", "ons_legal_status"):
        # Every in-frame firm has positive turnover and a populated cell; the
        # out-of-frame traders draw from their division's or group's row; no
        # firm reaches the UK total.
        assert int(by_level[(attribute, "cell")]) == in_frame
        assert int(by_level[(attribute, "row")]) == out_of_frame
        assert int(by_level[(attribute, "uk")]) == 0


def test_weighted_region_by_legal_status_tracks_held_out_table_15() -> None:
    held_out = pd.read_csv(
        FIXTURES / "firm_attributes_2025" / "ons_legal_status_region_turnover.csv",
        dtype={"region_code": str},
    )
    official = held_out.groupby(["region_code", "legal_status"])["enterprises"].sum()
    official = official / official.sum()

    firms = _with_attributes()
    frame = firms[firms["in_frame"].to_numpy(dtype=bool)]
    modelled = frame.groupby(["ons_region_code", "ons_legal_status"])[
        "firm_weight"
    ].sum()
    modelled.index = modelled.index.set_names(["region_code", "legal_status"])
    modelled = (modelled / modelled.sum()).reindex(official.index, fill_value=0.0)

    # Held out, not calibrated: region and legal status are drawn
    # independently given SIC x turnover cells, so the joint share is
    # loose. Locally (seed 42) the total variation distance is 0.033 and the
    # company share 0.767 against 0.767; gate only gross failures.
    assert float((modelled - official).abs().sum()) / 2 < 0.06
    company = modelled.xs(1, level="legal_status").sum()
    official_company = official.xs(1, level="legal_status").sum()
    assert abs(company - official_company) < 0.02


def test_paybill_export_tracks_ashe_jobs_times_mean_pay() -> None:
    """A coarse headcount check: weighted paybill near ASHE jobs x mean pay.

    Not calibrated: headcount x section mean, with proprietors and
    out-of-frame traders included. ASHE 2025's all-employees row gives
    24,889 thousand jobs at a mean of 40,444.9 pounds, about 1.007 trillion
    pounds; the band only catches gross failures, such as a headcount or
    unit error. Every firm here has a published section, so none falls back
    to the all-employees mean.
    """

    source = read_uk_firm_paybill_source(FIXTURES / "firm_paybill_2024")
    firms = _with_attributes()
    firms = firms.assign(firm_paybill=assign_uk_firm_paybill(firms, source))
    table = to_policyengine_uk_firm_table(
        firms, threshold_k=_generated().config.vat_threshold
    )

    assert tuple(table.columns) == UK_FIRM_PAYBILL_TABLE_COLUMNS
    validate_uk_firm_table(table)
    weighted = float((table["firm_weight"] * table["firm_paybill"]).sum())
    ashe = 24_889_000 * 40_444.9
    assert 0.85 < weighted / ashe < 1.25
    counts = uk_firm_paybill_fallback_counts(table, source).set_index("level")
    assert float(counts["weighted_paybill_gbp"].sum()) == pytest.approx(weighted)
    assert int(counts.loc["all_employees", "firms"]) == 0
