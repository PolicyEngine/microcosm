from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import load_puf_tax_unit_donor, puf_donor_io
from microcosm.build.us_runtime.puf_source_agi import (
    PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS,
)


def _write_processed_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as h5:
        for name, values in arrays.items():
            h5.create_dataset(name, data=values)


def test_loader_threads_source_alignment_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    processed_path = tmp_path / "processed.h5"
    source_path = tmp_path / "puf_2015.csv"
    source_path.write_text("restricted source is read by the alignment seam")
    arrays = {
        "tax_unit_id": np.asarray([10, 20], dtype=np.int64),
        "household_weight": np.asarray([1.0, 2.0]),
        "untouched": np.asarray([3, 4], dtype=np.int16),
    }
    _write_processed_arrays(processed_path, arrays)
    adjusted_gross_income = np.asarray([-5_000.0, 250_000.0])
    captured: dict[str, object] = {}

    def fake_agi(
        actual_source_path: Path,
        *,
        processed_tax_unit_ids,
        processed_tax_unit_weights,
    ) -> np.ndarray:
        captured["source_path"] = actual_source_path
        captured["processed_tax_unit_ids"] = processed_tax_unit_ids
        captured["processed_tax_unit_weights"] = processed_tax_unit_weights
        return adjusted_gross_income

    expected = pd.DataFrame({"tax_unit_id": [10, 20]})
    schedule_e = {
        "partnership_income": np.asarray([1_000.0, -50.0]),
        "passive_partnership_s_corp_income": np.asarray([200.0, 0.0]),
    }
    processed_nonpassive = np.asarray([900.0, -60.0])

    def fake_processed_nonpassive(actual_arrays) -> np.ndarray:
        captured["nonpassive_arrays"] = actual_arrays
        return processed_nonpassive

    def fake_schedule_e(
        actual_source_path: Path,
        *,
        processed_tax_unit_ids,
        processed_tax_unit_weights,
        processed_nonpassive_partnership_s_corp_income,
        receipt,
    ) -> dict[str, np.ndarray]:
        captured["schedule_e_source_path"] = actual_source_path
        captured["schedule_e_ids"] = processed_tax_unit_ids
        captured["schedule_e_weights"] = processed_tax_unit_weights
        captured["schedule_e_nonpassive"] = (
            processed_nonpassive_partnership_s_corp_income
        )
        receipt["weighted_line_32"] = 900.0
        return schedule_e

    def fake_donor(
        actual_arrays,
        *,
        adjusted_gross_income,
        schedule_e_partnership_s_corp,
        donor_build_summary,
    ) -> pd.DataFrame:
        captured["arrays"] = actual_arrays
        captured["adjusted_gross_income"] = adjusted_gross_income
        captured["schedule_e"] = schedule_e_partnership_s_corp
        captured["donor_build_summary"] = donor_build_summary
        donor_build_summary["mortgage_field_quarantine"] = {"screened_record_count": 2}
        return expected

    monkeypatch.setattr(
        puf_donor_io,
        "source_year_puf_adjusted_gross_income",
        fake_agi,
    )
    monkeypatch.setattr(
        puf_donor_io,
        "processed_puf_tax_unit_partnership_s_corp_income",
        fake_processed_nonpassive,
    )
    monkeypatch.setattr(
        puf_donor_io,
        "source_year_puf_partnership_s_corp_income",
        fake_schedule_e,
    )
    monkeypatch.setattr(
        puf_donor_io,
        "puf_tax_unit_donor_from_arrays",
        fake_donor,
    )

    summary: dict[str, object] = {}
    actual = load_puf_tax_unit_donor(
        processed_path,
        source_path,
        donor_build_summary=summary,
    )

    assert actual is expected
    assert captured["source_path"] == source_path
    np.testing.assert_array_equal(
        captured["processed_tax_unit_ids"],
        arrays["tax_unit_id"],
    )
    np.testing.assert_array_equal(
        captured["processed_tax_unit_weights"],
        arrays["household_weight"],
    )
    loaded_arrays = captured["arrays"]
    assert isinstance(loaded_arrays, dict)
    assert set(loaded_arrays) == set(arrays)
    for name, values in arrays.items():
        np.testing.assert_array_equal(loaded_arrays[name], values)
    assert captured["adjusted_gross_income"] is adjusted_gross_income
    assert captured["schedule_e_source_path"] == source_path
    np.testing.assert_array_equal(captured["schedule_e_ids"], arrays["tax_unit_id"])
    np.testing.assert_array_equal(
        captured["schedule_e_weights"], arrays["household_weight"]
    )
    assert captured["nonpassive_arrays"] is loaded_arrays
    assert captured["schedule_e_nonpassive"] is processed_nonpassive
    assert captured["schedule_e"] is schedule_e
    assert captured["donor_build_summary"] is summary
    assert summary == {
        "mortgage_field_quarantine": {"screened_record_count": 2},
        "schedule_e_line_32": {"weighted_line_32": 900.0},
    }


def test_loader_refuses_missing_source_year_path(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.h5"
    _write_processed_arrays(
        processed_path,
        {
            "tax_unit_id": np.asarray([1], dtype=np.int64),
            "household_weight": np.asarray([1.0]),
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "--puf-source-year-csv is required to align nonzero E19200 records "
            "to the published TY2015 SOI AGI bands"
        ),
    ):
        load_puf_tax_unit_donor(processed_path, None)


def test_loader_surfaces_source_year_recid_order_refusal(tmp_path: Path) -> None:
    processed_path = tmp_path / "processed.h5"
    _write_processed_arrays(
        processed_path,
        {
            "tax_unit_id": np.asarray([2, 1, 1_000_000], dtype=np.int64),
            "household_weight": np.asarray([1.0, 1.0, 1.0]),
        },
    )
    source_path = tmp_path / "puf_2015.csv"
    source = pd.DataFrame(
        {
            column: np.zeros(6, dtype=np.float64)
            for column in PUF_SOURCE_YEAR_AGI_REQUIRED_COLUMNS
        }
    )
    source["RECID"] = [1, 2, 999_996, 999_997, 999_998, 999_999]
    source["MARS"] = [1, 1, 0, 0, 0, 0]
    source["S006"] = 100.0
    source["E00100"] = [10_000, 20_000, -1, 1, 10_000_000, 100_000_000]
    source.to_csv(source_path, index=False)

    with pytest.raises(
        ValueError,
        match="Processed PUF regular RECID order does not match the TY2015 source",
    ):
        load_puf_tax_unit_donor(processed_path, source_path)
