"""Raw IRS PUF aging contracts and archived parity checks."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceOperationSpec
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from populace.build.us_runtime.puf_aging import (
    PUF_AGING_ARCHIVED_PROFILE_VERSION,
    PufAgingColumnFactor,
    age_raw_puf,
    load_archived_puf_aging_factors,
    puf_aging_factors_from_mapping,
)
from populace.build.us_runtime.source_runtime import (
    uprate_us_raw_puf_from_manifest,
    us_source_operation_handlers,
)

_RAW_PUF_2015_PATH = os.environ.get("POPULACE_RAW_PUF_2015_CSV")
_PUF_2024_PATH = os.environ.get("POPULACE_PUF_2024_H5")
requires_raw_artifacts = pytest.mark.skipif(
    not _RAW_PUF_2015_PATH
    or not Path(_RAW_PUF_2015_PATH).is_file()
    or not _PUF_2024_PATH
    or not Path(_PUF_2024_PATH).is_file(),
    reason=(
        "set POPULACE_RAW_PUF_2015_CSV and POPULACE_PUF_2024_H5 to "
        "the restricted local artifacts"
    ),
)


def _factor_payload(
    *,
    aging_version: str = "ledger_v1",
    source_year: int = 2015,
    target_year: int = 2024,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "aging_version": aging_version,
        "source_year": source_year,
        "target_year": target_year,
        "provenance": {
            "source_kind": "test_ledger",
            "source_artifact_sha256": "a" * 64,
            "source_coordinates": ["ledger:test-fact-bundle"],
            "notes": "Synthetic factors for a unit test.",
        },
        "weight": {
            "column": "S006",
            "factor": 1.1,
            "fact_ids": ["returns"],
        },
        "straight": [
            {
                "column": "amount",
                "factor": 2.0,
                "fact_ids": ["amount"],
            }
        ],
        "signed": [
            {
                "column": "business",
                "positive_factor": 3.0,
                "negative_factor": 4.0,
                "fact_ids": ["business-profit", "business-loss"],
            }
        ],
        "unchanged_columns": ["RECID"],
    }


def test_archived_profile_pins_effective_release_factors_and_no_2024_leg() -> None:
    factors = load_archived_puf_aging_factors()
    straight = {factor.column: factor.factor for factor in factors.straight}
    signed = {factor.column: factor for factor in factors.signed}

    assert factors.aging_version == PUF_AGING_ARCHIVED_PROFILE_VERSION
    assert (factors.source_year, factors.target_year) == (2015, 2021)
    assert factors.provenance.source_artifact_sha256 == (
        "0d60ddbe0b2c0a00d8c0521ab1693e51487595fb941314e114ec52901bba04bb"
    )
    assert factors.weight.factor == pytest.approx(1.0686481028722197)
    assert straight["E00200"] == pytest.approx(1.187079295246555)
    assert straight["E00400"] == pytest.approx(0.8396766287008147)
    assert straight["E19200"] == pytest.approx(1.02**6)
    assert signed["E00900"].positive_factor == pytest.approx(1.6738813405953035)
    assert signed["E00900"].negative_factor == pytest.approx(2.2268437882848242)
    assert signed["E26270"].positive_factor == pytest.approx(2.0764087584558091)
    assert signed["E26270"].negative_factor == pytest.approx(2.6139943101980907)
    assert "E02100" in factors.unchanged_columns
    assert "E87530" in factors.unchanged_columns


def test_pure_aging_applies_straight_signed_and_weight_factors() -> None:
    factors = puf_aging_factors_from_mapping(_factor_payload())
    source = pd.DataFrame(
        {
            "RECID": [1, 2, 3],
            "S006": [100, 200, 300],
            "amount": [1.0, 0.0, -2.0],
            "business": [5.0, 0.0, -7.0],
            "not_owned": [9, 8, 7],
        }
    )
    original = source.copy(deep=True)

    aged = age_raw_puf(source, factors=factors)

    pd.testing.assert_frame_equal(source, original)
    np.testing.assert_allclose(aged["S006"], [110.0, 220.0, 330.0])
    np.testing.assert_allclose(aged["amount"], [2.0, 0.0, -4.0])
    np.testing.assert_allclose(aged["business"], [15.0, 0.0, -28.0])
    pd.testing.assert_series_equal(aged["RECID"], source["RECID"])
    pd.testing.assert_series_equal(aged["not_owned"], source["not_owned"])
    provenance = aged.attrs["populace_puf_aging"]
    assert provenance["aging_version"] == "ledger_v1"
    assert provenance["source_year"] == 2015
    assert provenance["target_year"] == 2024
    assert provenance["row_count"] == 3
    assert provenance["fact_ids"] == [
        "amount",
        "business-loss",
        "business-profit",
        "returns",
    ]


def test_aging_rejects_duplicate_ownership_missing_columns_and_nonfinite_values() -> (
    None
):
    factors = puf_aging_factors_from_mapping(_factor_payload())
    duplicate = replace(
        factors,
        straight=(
            *factors.straight,
            PufAgingColumnFactor(
                column="business",
                factor=1.0,
                fact_ids=("duplicate",),
            ),
        ),
    )
    with pytest.raises(ValueError, match="multiple owners"):
        duplicate.validate()

    valid = pd.DataFrame(
        {
            "RECID": [1],
            "S006": [100.0],
            "amount": [1.0],
            "business": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing aging column"):
        age_raw_puf(valid.drop(columns="amount"), factors=factors)
    with pytest.raises(ValueError, match="nonfinite"):
        age_raw_puf(
            valid.assign(business=np.inf),
            factors=factors,
        )


def test_manifest_handler_is_registered_versioned_and_fail_closed() -> None:
    factors = puf_aging_factors_from_mapping(_factor_payload())
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "uprate",
            "from_year": 2015,
            "to_year_from_build_config": True,
            "aging_version": "ledger_v1",
            "factor_bundle_config_key": "puf_aging_factors",
        }
    )
    source = pd.DataFrame(
        {
            "RECID": [1],
            "S006": [100.0],
            "amount": [2.0],
            "business": [-3.0],
        }
    )
    context = SourceRuntimeContext(
        config=SourceRuntimeConfig(
            target_year=2024,
            extra={"puf_aging_factors": factors},
        ),
        tables={},
    )

    assert us_source_operation_handlers()["uprate"] is uprate_us_raw_puf_from_manifest
    result = uprate_us_raw_puf_from_manifest(source, operation, context)
    assert result["amount"].item() == 4.0
    assert result["business"].item() == -12.0

    with pytest.raises(SourceRuntimeError, match="explicit factor bundle"):
        uprate_us_raw_puf_from_manifest(
            source,
            operation,
            SourceRuntimeContext(
                config=SourceRuntimeConfig(target_year=2024),
                tables={},
            ),
        )
    with pytest.raises(SourceRuntimeError, match="target_year"):
        uprate_us_raw_puf_from_manifest(
            source,
            operation,
            SourceRuntimeContext(
                config=SourceRuntimeConfig(
                    target_year=2025,
                    extra={"puf_aging_factors": factors},
                ),
                tables={},
            ),
        )


@requires_raw_artifacts
def test_archived_profile_reproduces_release_aging_arithmetic() -> None:
    h5py = pytest.importorskip("h5py")
    assert _RAW_PUF_2015_PATH is not None
    assert _PUF_2024_PATH is not None
    factors = load_archived_puf_aging_factors()
    raw = pd.read_csv(
        _RAW_PUF_2015_PATH,
        usecols=list(factors.required_columns),
    )
    assert len(raw) == 207_696

    # Release 1.8.0 dropped the four disclosure aggregates instead of
    # disaggregating them. Some aggregate-only cells are NaN, so reproduce
    # that row-topology decision before the strict arithmetic pass.
    raw = raw.loc[raw["MARS"] != 0].copy()
    aged = age_raw_puf(raw, factors=factors)
    aged = aged.set_index("RECID")
    assert len(aged) == 207_692

    with h5py.File(_PUF_2024_PATH) as artifact:
        tax_unit_ids = artifact["tax_unit_id"][:]
        person_tax_unit_ids = artifact["person_tax_unit_id"][:]
        assert np.all(tax_unit_ids[:-1] <= tax_unit_ids[1:])
        aligned = aged.reindex(tax_unit_ids)
        assert not aligned.isna().any(axis=None)

        np.testing.assert_allclose(
            artifact["household_weight"][:],
            aligned["S006"].to_numpy() / 100.0,
            rtol=1e-13,
            atol=1e-10,
        )

        def grouped_person_array(column: str) -> np.ndarray:
            grouped = (
                pd.DataFrame(
                    {
                        "tax_unit_id": person_tax_unit_ids,
                        "value": artifact[column][:],
                    }
                )
                .groupby("tax_unit_id", sort=False)["value"]
                .sum()
            )
            return grouped.reindex(tax_unit_ids).to_numpy()

        expected = {
            "employment_income": aligned["E00200"].to_numpy(),
            "taxable_interest_income": aligned["E00300"].to_numpy(),
            "tax_exempt_interest_income": aligned["E00400"].to_numpy(),
            "qualified_dividend_income": aligned["E00650"].to_numpy(),
            "non_qualified_dividend_income": (
                aligned["E00600"] - aligned["E00650"]
            ).to_numpy(),
            "taxable_ira_distributions": aligned["E01400"].to_numpy(),
            "taxable_pension_income": aligned["E01700"].to_numpy(),
            "social_security": aligned["E02400"].to_numpy(),
            "short_term_capital_gains": aligned["P22250"].to_numpy(),
            "long_term_capital_gains": aligned["P23250"].to_numpy(),
            "farm_rent_income": aligned["E27200"].to_numpy(),
            "estate_income": (aligned["E26390"] - aligned["E26400"]).to_numpy(),
            "rental_income": (aligned["E25850"] - aligned["E25860"]).to_numpy(),
            "self_employment_income": aligned["E00900"].to_numpy(),
            "partnership_s_corp_income": aligned["E26270"].to_numpy(),
            # The physical release used E03230 only. The later E87530 max
            # fallback is not embedded in these bytes.
            "qualified_tuition_expenses": aligned["E03230"].to_numpy(),
        }
        for column, values in expected.items():
            np.testing.assert_allclose(
                grouped_person_array(column),
                values,
                rtol=1e-12,
                atol=1e-7,
            )

        expected_w2 = 0.16 * np.maximum(
            0.0,
            aligned["E00900"]
            + aligned["E26270"]
            + aligned["E02100"]
            + aligned["E27200"],
        )
        np.testing.assert_allclose(
            grouped_person_array("w2_wages_from_qualified_business"),
            expected_w2,
            rtol=1e-12,
            atol=1e-7,
        )
