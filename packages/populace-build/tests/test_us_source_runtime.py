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
from populace.build.us import US_SOURCE_MANIFEST
from populace.build.us.puf_aggregate_records import (
    AGGREGATE_RECIDS,
    SYNTHETIC_RECID_START,
    disaggregate_puf_aggregate_records,
)
from populace.build.us.source_runtime import (
    disaggregate_us_puf_aggregate_records_from_manifest,
    us_source_operation_handlers,
)


def _make_runtime_mini_puf() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows: list[dict[str, float | int]] = []
    donor_specs = [
        (999996, -18_000_000.0, -500_000.0),
        (999997, 250_000.0, 9_500_000.0),
        (999998, 12_000_000.0, 90_000_000.0),
        (999999, 125_000_000.0, 650_000_000.0),
    ]
    next_recid = 1
    for _bucket_recid, low, high in donor_specs:
        for index in range(25):
            agi = float(rng.uniform(low, high))
            sign = -1.0 if agi < 0 else 1.0
            abs_agi = abs(agi)
            rows.append(
                {
                    "RECID": next_recid,
                    "S006": 100,
                    "MARS": 2 if index % 3 == 0 else 1,
                    "XTOT": 2 if index % 3 == 0 else 1,
                    "DSI": 0,
                    "EIC": index % 4,
                    "E00100": agi,
                    "E00200": abs_agi * 0.08,
                    "P23250": abs_agi * rng.uniform(0.05, 0.35) * sign,
                    "P22250": abs_agi * rng.uniform(0.01, 0.12) * sign,
                    "E00650": abs_agi * 0.04,
                    "E00300": abs_agi * 0.03,
                    "E26270": abs_agi * rng.uniform(0.01, 0.15) * sign,
                    "E00900": abs_agi * 0.03 * sign,
                    "E02100": abs_agi * 0.01 * sign,
                    "E00400": abs_agi * 0.01,
                    "E00600": abs_agi * 0.05,
                }
            )
            next_recid += 1

    aggregate_rows = [
        (999996, -5_000_000.0),
        (999997, 5_000_000.0),
        (999998, 30_000_000.0),
        (999999, 300_000_000.0),
    ]
    for recid, agi in aggregate_rows:
        sign = -1.0 if agi < 0 else 1.0
        abs_agi = abs(agi)
        rows.append(
            {
                "RECID": recid,
                "S006": 2_000,
                "MARS": 0,
                "XTOT": 1,
                "DSI": 0,
                "EIC": 0,
                "E00100": agi,
                "E00200": abs_agi * 0.08,
                "P23250": abs_agi * 0.30 * sign,
                "P22250": abs_agi * 0.08 * sign,
                "E00650": abs_agi * 0.04,
                "E00300": abs_agi * 0.03,
                "E26270": abs_agi * 0.10 * sign,
                "E00900": abs_agi * 0.03 * sign,
                "E02100": abs_agi * 0.01 * sign,
                "E00400": abs_agi * 0.01,
                "E00600": abs_agi * 0.05,
            }
        )
    return pd.DataFrame(rows)


def test_us_puf_manifest_prefix_runs_aggregate_disaggregation() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
    mini_puf = _make_runtime_mini_puf()

    result = run_source_stage(
        stage,
        tables={"puf_tax_unit": mini_puf},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="disaggregate_aggregate_records",
    )

    expected = disaggregate_puf_aggregate_records(mini_puf, seed=42)
    pd.testing.assert_frame_equal(result, expected)
    assert not result["RECID"].isin(AGGREGATE_RECIDS).any()
    assert (result["RECID"] >= SYNTHETIC_RECID_START).any()


def test_us_puf_manifest_prefix_uses_build_seed() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
    mini_puf = _make_runtime_mini_puf()

    first = run_source_stage(
        stage,
        tables={"puf_tax_unit": mini_puf},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=1, target_year=2024),
        stop_after="disaggregate_aggregate_records",
    )
    second = run_source_stage(
        stage,
        tables={"puf_tax_unit": mini_puf},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=2, target_year=2024),
        stop_after="disaggregate_aggregate_records",
    )

    assert not first.equals(second)


def test_us_puf_handler_validates_packaged_spec_shape() -> None:
    mini_puf = _make_runtime_mini_puf()
    bad_operation = SourceOperationSpec.from_mapping(
        {
            "kind": "disaggregate_aggregate_records",
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
        }
    )

    with pytest.raises(SourceRuntimeError, match="replace_records"):
        disaggregate_us_puf_aggregate_records_from_manifest(
            mini_puf,
            bad_operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )


def test_us_puf_handler_rejects_unknown_parameters() -> None:
    mini_puf = _make_runtime_mini_puf()
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "disaggregate_aggregate_records",
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999996, 999997, 999998, 999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
            "seed_from_build_config": True,
            "seed": 7,
        }
    )

    with pytest.raises(SourceRuntimeError, match="unsupported parameter"):
        disaggregate_us_puf_aggregate_records_from_manifest(
            mini_puf,
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )


def test_us_puf_handler_requires_build_seed() -> None:
    mini_puf = _make_runtime_mini_puf()
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "disaggregate_aggregate_records",
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999996, 999997, 999998, 999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
            "seed_from_build_config": False,
        }
    )

    with pytest.raises(SourceRuntimeError, match="seed_from_build_config=true"):
        disaggregate_us_puf_aggregate_records_from_manifest(
            mini_puf,
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )
