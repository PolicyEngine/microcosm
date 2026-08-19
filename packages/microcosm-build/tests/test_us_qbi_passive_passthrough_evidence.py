"""Synthetic and packaged-resource tests for passive pass-through evidence."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.qbi_passive_passthrough_evidence import (
    FORM_8960_LINE_4C_AMOUNT,
    FORM_8960_LINE_4C_TO_4A_RATIO,
    SCF_PASSIVE_INCOME_BANDS,
    SCF_PASSIVE_REQUIRED_COLUMNS,
    build_qbi_passive_passthrough_resource,
    build_scf_passive_passthrough_records,
    load_qbi_passive_passthrough_resource,
    validate_qbi_passive_passthrough_resource,
    weighted_inverse_cdf,
)

ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "tools/build_us_qbi_passive_passthrough_evidence.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "qbi_passive_passthrough_evidence_builder", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def _row(
    household_id: int,
    implicate: int,
    *,
    schedule_e_income: float,
    holder: bool,
    nonactive_value: float,
    active_value: float = 100.0,
    weight: float = 100.0,
) -> dict[str, float]:
    row = {column: 0.0 for column in SCF_PASSIVE_REQUIRED_COLUMNS}
    row.update(
        {
            "y1": float(household_id * 10 + implicate),
            "x42001": weight,
            "x3103": 1.0,
            "x3104": 1.0,
            "x3105": 1.0,
            "x3401": 1.0 if holder else 5.0,
            "x5714": schedule_e_income,
            "x3129": active_value,
            "x3408": nonactive_value,
        }
    )
    return row


def _scf_fixture() -> pd.DataFrame:
    band_values = (-1.0, 10_000.0, 50_000.0, 150_000.0, 500_000.0, 2_000_000.0)
    rows: list[dict[str, float]] = []
    household_id = 1
    for band_index, income in enumerate(band_values):
        # Every prevalence denominator meets n=30.  The first share cell meets
        # n=30; the other five each have one holder and must pool to all bands.
        for household_index in range(30):
            holder = band_index == 0 or household_index == 0
            nonactive = (
                float(10 + household_index)
                if holder and not (band_index == 0 and household_index == 0)
                else 0.0
            )
            for implicate in range(1, 6):
                rows.append(
                    _row(
                        household_id,
                        implicate,
                        schedule_e_income=income,
                        holder=holder,
                        nonactive_value=nonactive,
                        weight=float(100 + household_index),
                    )
                )
            household_id += 1
    return pd.DataFrame.from_records(rows)


def _provenance() -> dict[str, object]:
    return {
        "generated_by": "synthetic test",
        "run_command": "synthetic fixture",
        "inputs": [{"filename": "synthetic.dta", "sha256": "0" * 64}],
        "deterministic": True,
        "random_draws": "none",
    }


def test_scf_balance_sheet_formulas_match_scfp_construction() -> None:
    rows = [
        _row(
            1,
            implicate,
            schedule_e_income=10_000.0,
            holder=True,
            nonactive_value=0.0,
        )
        for implicate in range(1, 6)
    ]
    source = pd.DataFrame.from_records(rows)
    source["x3129"] = 100.0
    source["x3124"] = 20.0
    source["x3126"] = 10.0
    source["x3127"] = 5.0
    source["x3121"] = 7.0
    source["x3122"] = 1.0
    source["x3229"] = 50.0
    source["x3224"] = 5.0
    source["x3226"] = 2.0
    source["x3227"] = 5.0
    source["x3221"] = 3.0
    source["x3222"] = 6.0
    source["x3335"] = 4.0
    source["x507"] = 10_000.0  # SCFP caps the business percentage at 90%.
    source["x513"] = 100.0
    source["x526"] = 20.0
    source["x805"] = 10.0
    source["x905"] = 5.0
    source["x1005"] = 5.0
    source["x1103"] = 1.0
    source["x1108"] = 10.0
    source["x1114"] = 5.0
    source["x1119"] = 20.0
    source["x1125"] = 1.0
    source["x1130"] = 30.0
    source["x1136"] = 12.0
    source["x3408"] = 10.0
    source["x3412"] = 20.0
    source["x3416"] = 30.0

    records = build_scf_passive_passthrough_records(source)

    # First active business 117; second 56; remaining businesses 4.
    # The direct farm-secured allocations leave reduced balances 1, 20, and
    # 3.  X1136 therefore assigns 1/6 to the farm: FARMBUS is
    # 90 - 9 - 27 - 1.8 = 52.2.  NONACTBUS = 60.
    assert records["actbus"].tolist() == pytest.approx([229.2] * 5)
    assert records["nonactbus"].tolist() == pytest.approx([60.0] * 5)
    assert records["nonactive_business_value_share"].tolist() == pytest.approx(
        [60.0 / 289.2] * 5
    )


def test_pooled_implicates_and_independent_thin_cell_fallbacks() -> None:
    payload = build_qbi_passive_passthrough_resource(
        _scf_fixture(), provenance=_provenance()
    )

    assert [cell["income_band"] for cell in payload["cells"]] == list(
        SCF_PASSIVE_INCOME_BANDS
    )
    first = payload["cells"][0]
    assert first["requested_counts"]["active_pooled_record_count"] == 150
    assert first["requested_counts"]["active_implicate_adjusted_unweighted_n"] == 30
    assert first["holding_prevalence"]["estimate_level"] == "exact"
    assert first["holding_prevalence"]["estimate"] == 1.0
    assert first["conditional_share"]["estimate_level"] == "exact"
    assert first["conditional_share"]["source_income_band"] == "nonpositive"

    thin = payload["cells"][1]
    assert thin["holding_prevalence"]["estimate_level"] == "exact"
    assert thin["conditional_share"]["estimate_level"] == "all_income_bands"
    assert thin["conditional_share"]["source_income_band"] == "all"
    assert (
        thin["requested_counts"]["nonactive_holder_implicate_adjusted_unweighted_n"]
        == 1.0
    )
    assert (
        thin["conditional_share"]["selected_quantiles"]
        == payload["cells"][2]["conditional_share"]["selected_quantiles"]
    )


def test_holding_screener_retains_reported_zero_value_holders() -> None:
    records = build_scf_passive_passthrough_records(_scf_fixture())
    payload = build_qbi_passive_passthrough_resource(
        _scf_fixture(), provenance=_provenance()
    )
    first = payload["cells"][0]

    assert first["requested_counts"]["nonactive_holder_pooled_record_count"] == 150
    first_band_holders = records.loc[
        records["income_band"].eq("nonpositive") & records["holds_nonactive_business"]
    ]
    assert first_band_holders["nonactive_business_value_share"].eq(0.0).sum() == 5


def test_weighted_inverse_cdf_uses_weights_and_stable_ties() -> None:
    quantiles = weighted_inverse_cdf(
        np.array([0.1, 0.2, 0.9]),
        np.array([1.0, 3.0, 1.0]),
    )

    assert quantiles == {
        "q05": 0.1,
        "q25": 0.2,
        "q50": 0.2,
        "q75": 0.2,
        "q95": 0.9,
    }


def test_builder_reads_equivalent_dta_and_zip_sources(tmp_path: Path) -> None:
    source = _scf_fixture().iloc[:30].copy()
    dta_path = tmp_path / "p22i6.dta"
    zip_path = tmp_path / "scf2022s.zip"
    source.to_stata(dta_path, write_index=False)
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(dta_path, arcname="p22i6.dta")
    builder = _load_builder()

    dta_frame, dta_inputs = builder.read_scf_source(dta_path)
    zip_frame, zip_inputs = builder.read_scf_source(zip_path)

    pd.testing.assert_frame_equal(dta_frame, zip_frame)
    assert [record["role"] for record in dta_inputs] == ["scf_2022_source"]
    assert [record["role"] for record in zip_inputs] == [
        "scf_2022_source",
        "scf_2022_archive_member",
    ]
    assert zip_inputs[1]["sha256"] == dta_inputs[0]["sha256"]


def test_schema_rejects_deprovisionalized_or_inconsistent_payload() -> None:
    payload = build_qbi_passive_passthrough_resource(
        _scf_fixture(), provenance=_provenance()
    )
    broken = deepcopy(payload)
    broken["provisional"] = False
    with pytest.raises(ValueError, match="must remain provisional"):
        validate_qbi_passive_passthrough_resource(broken)

    broken = deepcopy(payload)
    broken["cells"][1]["conditional_share"]["estimate_level"] = "exact"
    with pytest.raises(ValueError, match="share fallback"):
        validate_qbi_passive_passthrough_resource(broken)

    broken = deepcopy(payload)
    broken["external_anchor"]["passive_passthrough_bounds"]["upper"]["amount"] = 1.0
    with pytest.raises(ValueError, match="upper bound"):
        validate_qbi_passive_passthrough_resource(broken)


def test_packaged_resource_has_reviewed_cells_and_anchor() -> None:
    payload = load_qbi_passive_passthrough_resource()

    estimates = {
        cell["income_band"]: cell["holding_prevalence"]["estimate"]
        for cell in payload["cells"]
    }
    assert estimates == pytest.approx(
        {
            "nonpositive": 0.033492918161629,
            "0_to_25k": 0.041323551642334,
            "25k_to_100k": 0.064532919876429,
            "100k_to_250k": 0.096543423809845,
            "250k_to_1m": 0.212268790432713,
            "over_1m": 0.389806347521960,
        }
    )
    anchor = payload["external_anchor"]
    assert anchor["form_8960"]["line_4c"]["amount"] == FORM_8960_LINE_4C_AMOUNT
    assert anchor["line_4c_to_4a_survival_ratio"] == pytest.approx(
        FORM_8960_LINE_4C_TO_4A_RATIO
    )
    assert anchor["passive_passthrough_bounds"]["lower"]["amount"] == 0.0
    assert (
        anchor["passive_passthrough_bounds"]["upper"]["amount"]
        == FORM_8960_LINE_4C_AMOUNT
    )
    assert json.dumps(payload, allow_nan=False)
