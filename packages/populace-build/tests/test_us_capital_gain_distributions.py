from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceOperationSpec
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime import US_SOURCE_MANIFEST
from populace.build.us_runtime.capital_gain_distributions import (
    load_capital_gain_distribution_shares,
    split_us_component_by_share_from_manifest,
)
from populace.build.us_runtime.source_runtime import us_source_operation_handlers


def _toy_tax_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tax_unit_id": [1, 2, 3, 4, 5],
            "tax_unit_weight": [100.0, 100.0, 100.0, 100.0, 100.0],
            "long_term_capital_gains_before_response": [
                10_000.0,
                0.0,
                -4_000.0,
                50_000.0,
                8_000.0,
            ],
            "non_sch_d_capital_gains": [0.0, 0.0, 0.0, 0.0, 3_000.0],
        }
    )


def _split_operation(**overrides) -> SourceOperationSpec:
    raw = {
        "kind": "split_component_by_share",
        "resource": "soca_capital_gain_distribution_shares",
        "share_field": "schedule_d_cgd_share_of_lt_net_gains",
        "source_column": "long_term_capital_gains_before_response",
        "output": "schedule_d_capital_gain_distributions",
        "exclusive_with": ["non_sch_d_capital_gains"],
    }
    raw.update(overrides)
    return SourceOperationSpec.from_mapping(raw)


def _context() -> SourceRuntimeContext:
    return SourceRuntimeContext(config=SourceRuntimeConfig(), tables={})


class TestSharesResource:
    def test_packaged_resource_rederives_its_own_anchor(self) -> None:
        shares = load_capital_gain_distribution_shares()
        share = shares.schedule_d_cgd_share_of_lt_net_gains
        anchor = shares.anchor
        residual = (
            anchor["soca_all_route_capital_gain_distributions_k"]
            - anchor["pub1304_direct_1040_route_k"]
        )
        assert np.isclose(residual, anchor["schedule_d_route_residual_k"])
        assert np.isclose(
            share,
            residual
            / (
                anchor["soca_long_term_total_net_gain_k"]
                - anchor["pub1304_direct_1040_route_k"]
            ),
            rtol=1e-6,
        )
        # TY2015: $68.1B Schedule-D route over $690.7B of Schedule-D LT
        # net gains — the split is a bit under ten percent.
        assert 0.09 < share < 0.11


class TestCapitalGainDistributionsStage:
    def test_manifest_stage_splits_the_schedule_d_route(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()["capital_gain_distributions"]
        toy = _toy_tax_units()

        result = run_source_stage(
            stage,
            tables={"tax_unit": toy},
            operation_handlers=us_source_operation_handlers(),
            config=SourceRuntimeConfig(seed=0, target_year=2024),
        )

        share = (
            load_capital_gain_distribution_shares()
            .schedule_d_cgd_share_of_lt_net_gains
        )
        out = result["schedule_d_capital_gain_distributions"]
        assert np.isclose(out.iloc[0], 10_000.0 * share)
        # Zero and negative long-term gains get nothing.
        assert out.iloc[1] == 0.0
        assert out.iloc[2] == 0.0
        assert np.isclose(out.iloc[3], 50_000.0 * share)
        # The two reporting routes are mutually exclusive on a real return.
        assert out.iloc[4] == 0.0
        # The source column is untouched and the split never exceeds it.
        pd.testing.assert_series_equal(
            result["long_term_capital_gains_before_response"],
            toy["long_term_capital_gains_before_response"],
        )
        assert (
            out
            <= result["long_term_capital_gains_before_response"].clip(lower=0.0)
            + 1e-9
        ).all()
        # National reconstruction: the split is exactly proportional.
        eligible_lt = toy.loc[
            (toy["long_term_capital_gains_before_response"] > 0)
            & ~(toy["non_sch_d_capital_gains"] > 0),
            "long_term_capital_gains_before_response",
        ].sum()
        assert np.isclose(out.sum(), share * eligible_lt)

    def test_split_requires_a_frame(self) -> None:
        with pytest.raises(SourceRuntimeError, match="must follow read_table"):
            split_us_component_by_share_from_manifest(
                None, _split_operation(), _context()
            )

    def test_split_rejects_unknown_parameters(self) -> None:
        with pytest.raises(SourceRuntimeError, match="unexpected parameter"):
            split_us_component_by_share_from_manifest(
                _toy_tax_units(), _split_operation(bogus=1), _context()
            )

    def test_split_rejects_unknown_resource_and_share_field(self) -> None:
        with pytest.raises(SourceRuntimeError, match="knows only"):
            split_us_component_by_share_from_manifest(
                _toy_tax_units(),
                _split_operation(resource="somewhere_else"),
                _context(),
            )
        with pytest.raises(SourceRuntimeError, match="share_field"):
            split_us_component_by_share_from_manifest(
                _toy_tax_units(),
                _split_operation(share_field="not_a_field"),
                _context(),
            )

    def test_split_refuses_to_overwrite_an_existing_output(self) -> None:
        frame = _toy_tax_units()
        frame["schedule_d_capital_gain_distributions"] = 1.0
        with pytest.raises(SourceRuntimeError, match="already exists"):
            split_us_component_by_share_from_manifest(
                frame, _split_operation(), _context()
            )

    def test_split_requires_source_and_exclusive_columns(self) -> None:
        missing_source = _toy_tax_units().drop(
            columns=["long_term_capital_gains_before_response"]
        )
        with pytest.raises(SourceRuntimeError, match="source column"):
            split_us_component_by_share_from_manifest(
                missing_source, _split_operation(), _context()
            )
        missing_exclusive = _toy_tax_units().drop(
            columns=["non_sch_d_capital_gains"]
        )
        with pytest.raises(SourceRuntimeError, match="exclusive_with"):
            split_us_component_by_share_from_manifest(
                missing_exclusive, _split_operation(), _context()
            )

    def test_split_handler_is_registered(self) -> None:
        handlers = us_source_operation_handlers()
        assert (
            handlers["split_component_by_share"]
            is split_us_component_by_share_from_manifest
        )
