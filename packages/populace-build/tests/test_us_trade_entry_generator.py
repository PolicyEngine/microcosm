"""Tests for the synthetic import-entry generator (populace#615 P2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.us_trade.entry_generator import (
    EntrySizeAssumption,
    dotted_hts,
    generate_entries,
    solve_size_assumption,
    validate_entries_against_margins,
)


def _assumption(**overrides) -> EntrySizeAssumption:
    base = {
        "mean_entry_value": 100.0,
        "informal_count_share_target": 0.6,
        "informal_value_threshold": 2_500,
        "strata": 7,
        "sigma": 1.5,
        "achieved_informal_count_share": 0.6,
        "achieved_mean_entry_value": 100.0,
        "total_weighted_entries": 0,
        "anchor_provenance": {},
    }
    base.update(overrides)
    return EntrySizeAssumption(**base)


def _margins(rows) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["period", "hts10", "cty_code", "iso2", "con_val_mo"]
    )
    frame["chapter"] = frame["hts10"].str[:2]
    return frame


def test_dotted_hts_is_the_spine_format():
    assert dotted_hts("7202111000") == "7202.11.10.00"
    assert dotted_hts("0101210010") == "0101.21.00.10"
    with pytest.raises(ValueError, match="ten digits"):
        dotted_hts("720211100")
    with pytest.raises(ValueError, match="ten digits"):
        dotted_hts("72021110xx")


def test_every_cell_reproduced_exactly_on_randomized_margins():
    rng = np.random.default_rng(20260804)
    values = np.concatenate(
        [
            rng.integers(1, 10, 40),
            rng.integers(10, 5_000, 40),
            rng.integers(5_000, 50_000_000, 40),
            np.array([1, 2, 3, 7, 999_999_937, 10**12]),
        ]
    )
    rows = [
        (
            f"2026-{1 + (index % 2):02d}",
            f"{index % 97 + 1:02d}01210010",
            f"{1000 + index}",
            "CA",
            int(value),
        )
        for index, value in enumerate(values)
    ]
    margins = _margins(rows)
    entries = generate_entries(margins, _assumption(sigma=2.8, mean_entry_value=500.0))
    report = validate_entries_against_margins(entries, margins)
    assert report["exact"] is True
    assert report["cells_checked"] == len(margins)
    assert report["weighted_customs_value"] == int(margins["con_val_mo"].sum())
    assert (entries["customs_value"] >= 1).all()
    assert (entries["weight"] >= 1).all()


def test_small_cells_become_single_full_value_entries():
    margins = _margins(
        [
            ("2026-01", "0101210010", "1220", "CA", 1),
            ("2026-01", "0101210010", "2010", "MX", 57),
        ]
    )
    entries = generate_entries(margins, _assumption(mean_entry_value=32_918.0))
    assert len(entries) == 2
    assert entries["weight"].tolist() == [1, 1]
    assert sorted(entries["customs_value"].tolist()) == [1, 57]
    assert entries["entry_id"].str.startswith("synthetic-").all()


def test_large_cell_gets_increasing_strata_and_exact_weights():
    margins = _margins([("2026-01", "8471300100", "5700", "CN", 1_000_000)])
    assumption = _assumption(mean_entry_value=1_000.0, sigma=2.0)
    entries = generate_entries(margins, assumption)
    assert entries["weight"].sum() == 1_000  # round(V / mean)
    assert int((entries["weight"] * entries["customs_value"]).sum()) == 1_000_000
    assert entries["size_stratum"].nunique() == 7
    stratum_means = (entries["weight"] * entries["customs_value"]).groupby(
        entries["size_stratum"]
    ).sum() / entries["weight"].groupby(entries["size_stratum"]).sum()
    assert stratum_means.is_monotonic_increasing
    assert entries["hts_number"].unique().tolist() == ["8471.30.01.00"]


def test_generation_is_deterministic():
    margins = _margins(
        [
            ("2026-01", "8471300100", "5700", "CN", 123_456_789),
            ("2026-02", "0101210010", "1220", "CA", 4_321),
        ]
    )
    assumption = _assumption(mean_entry_value=777.0, sigma=2.3)
    first = generate_entries(margins, assumption)
    second = generate_entries(margins, assumption)
    pd.testing.assert_frame_equal(first, second)


def test_sigma_solver_hits_reachable_informal_share():
    rng = np.random.default_rng(7)
    cells = rng.integers(1_000, 10_000_000, 500)
    assumption = solve_size_assumption(
        cells,
        mean_entry_value=30_000.0,
        informal_count_share=0.65,
        anchor_provenance={"basis": "test"},
    )
    assert abs(assumption.achieved_informal_count_share - 0.65) < 0.02
    assert 0.05 <= assumption.sigma <= 6.0
    assert assumption.total_weighted_entries > 0
    register = assumption.register()
    assert register["synthetic"] is True
    assert "19 CFR 143.21" in register["statement"]
    assert register["known_gaps"]["is_postal_shipment"]


def test_sigma_solver_records_boundary_when_target_unreachable():
    cells = np.full(50, 10_000_000, dtype=np.int64)
    assumption = solve_size_assumption(
        cells,
        mean_entry_value=5_000_000.0,
        informal_count_share=0.99,
    )
    assert assumption.sigma == 6.0
    assert assumption.achieved_informal_count_share < 0.99


def test_generate_skips_zero_value_cells_and_requires_columns():
    margins = _margins(
        [
            ("2026-01", "0101210010", "1220", "CA", 5_000),
            ("2026-01", "0101210010", "2010", "MX", 0),
        ]
    )
    entries = generate_entries(margins, _assumption())
    assert entries["census_country_code"].unique().tolist() == ["1220"]
    with pytest.raises(ValueError, match="missing column"):
        generate_entries(margins.drop(columns=["iso2"]), _assumption())
    with pytest.raises(ValueError, match="positive consumption value"):
        generate_entries(margins.loc[margins["con_val_mo"] == 0], _assumption())


def test_schema_contract_and_synthetic_labeling():
    margins = _margins([("2026-03", "9903810100", "5700", "CN", 44_000)])
    entries = generate_entries(margins, _assumption(mean_entry_value=10_000.0))
    assert entries["entry_date"].tolist() == [pd.Timestamp("2026-03-15").date()] * len(
        entries
    )
    assert (entries["shipment_value"] == entries["customs_value"]).all()
    assert (~entries["is_postal_shipment"]).all()
    assert entries["country_of_origin"].unique().tolist() == ["CN"]
    assert entries["hts_number"].str.fullmatch(r"\d{4}\.\d{2}\.\d{2}\.\d{2}").all()
    assert entries["entry_id"].is_unique


def test_validation_catches_tampering():
    margins = _margins(
        [
            ("2026-01", "0101210010", "1220", "CA", 9_999),
            ("2026-01", "8471300100", "5700", "CN", 1_000_000),
        ]
    )
    entries = generate_entries(margins, _assumption(mean_entry_value=1_000.0))
    validate_entries_against_margins(entries, margins)

    tampered = entries.copy()
    tampered.loc[0, "customs_value"] += 1
    with pytest.raises(ValueError, match="!= margin"):
        validate_entries_against_margins(tampered, margins)

    postal = entries.copy()
    postal.loc[0, "is_postal_shipment"] = True
    with pytest.raises(ValueError, match="uniformly False"):
        validate_entries_against_margins(postal, margins)

    dropped = entries.iloc[1:]
    with pytest.raises(ValueError, match="Cell count mismatch|do not match"):
        validate_entries_against_margins(
            dropped.loc[dropped["census_country_code"] == "1220"], margins
        )


def test_weight_times_value_never_overflows_quietly():
    margins = _margins([("2026-01", "8471300100", "5700", "CN", 10**12)])
    entries = generate_entries(margins, _assumption(mean_entry_value=30_000.0))
    total = int((entries["weight"].astype(object) * entries["customs_value"]).sum())
    assert total == 10**12
    report = validate_entries_against_margins(entries, margins)
    assert report["weighted_customs_value"] == 10**12
