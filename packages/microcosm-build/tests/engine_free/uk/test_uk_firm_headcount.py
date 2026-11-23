"""UK firm headcount: proprietors, employees and the BPE small-firm redraw."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import firm_headcount
from microcosm.build.uk_runtime.firm_headcount import (
    UK_FIRM_BPE_LEGAL_STATUS_COLUMNS,
    UK_FIRM_BPE_LEGAL_STATUS_FILE,
    UK_FIRM_BPE_SIZE_BANDS_COLUMNS,
    UK_FIRM_BPE_SIZE_BANDS_FILE,
    UK_FIRM_EMPLOYEE_SIZE_BANDS,
    UK_FIRM_HEADCOUNT_STATUS,
    UK_FIRM_ONS_SMALL_BAND,
    UK_FIRM_SMALL_HEADCOUNT_STATUSES,
    UKFirmBPESource,
    fit_truncated_geometric,
    read_uk_firm_bpe_source,
    redraw_uk_firm_small_headcount,
    uk_firm_assumed_proprietors,
    uk_firm_bpe_size_band_reconciliation,
    uk_firm_employee_size_bands,
    uk_firm_employees,
    uk_firm_small_headcount_diagnostics,
    uk_firm_small_headcount_fits,
    uk_firm_small_headcount_targets,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_bpe_2025"
PROVENANCE = FIXTURE / "firm_bpe_2025.provenance.json"
TOOL = PATHS.repository / "tools" / "extract_uk_firm_bpe_tables.py"
MODULE = (
    PATHS.package / "src" / "microcosm" / "build" / "uk_runtime" / ("firm_headcount.py")
)
WORKBOOK_SHA256 = "223790dfd61d41d5866b747f2ea24535cef0db754ee261ae9a2d88bba9f478d7"


@cache
def _source() -> UKFirmBPESource:
    return read_uk_firm_bpe_source(FIXTURE)


@cache
def _provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


@cache
def _load_tool():
    name = "extract_uk_firm_bpe_tables"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _table_3(section: str, band: str) -> tuple[float, float]:
    frame = _source().legal_status
    row = frame[(frame["legal_status"] == section) & (frame["size_band"] == band)]
    assert len(row) == 1
    return (
        float(row["business_number"].iloc[0]),
        float(row["employment_thousand"].iloc[0]) * 1000,
    )


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    """A synthetic generator frame: every status, employment 1 to 300."""

    rng = np.random.default_rng(seed)
    status = rng.integers(1, 8, n)
    employment = np.where(
        rng.random(n) < 0.7, rng.integers(1, 5, n), rng.integers(5, 300, n)
    )
    ids = rng.permutation(n).astype(np.int64) + 1
    return pd.DataFrame(
        {
            "firm_id": ids,
            "source_firm_id": ids,
            "ons_legal_status": status.astype(np.int64),
            "employment": employment.astype(int),
            "firm_weight": rng.uniform(0.5, 5.0, n).astype(np.float32),
            "other": rng.random(n),
        }
    )


# -- fixture -------------------------------------------------------------------


def test_fixture_hashes_match_provenance() -> None:
    record = _provenance()
    assert record["source"]["sha256"] == WORKBOOK_SHA256
    assert "none was filled from memory" in record["method"]
    assert record["accessed"] == "2026-11-22"
    for name in (UK_FIRM_BPE_LEGAL_STATUS_FILE, UK_FIRM_BPE_SIZE_BANDS_FILE):
        payload = (FIXTURE / name).read_bytes()
        entry = record["files"][name]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert _source().sha256[name] == entry["sha256"]
        assert len(payload) == entry["bytes"]


def test_fixture_reads_with_columns_and_suppression() -> None:
    source = _source()
    assert tuple(source.legal_status.columns) == UK_FIRM_BPE_LEGAL_STATUS_COLUMNS
    assert tuple(source.size_bands.columns) == UK_FIRM_BPE_SIZE_BANDS_COLUMNS
    assert set(source.legal_status["legal_status"]) == {
        "private_sector",
        "companies",
        "ordinary_partnerships",
        "sole_proprietorships",
    }
    suppressed = source.legal_status["employment_flag"] == "[c]"
    assert suppressed.any()
    assert source.legal_status.loc[suppressed, "employment_thousand"].isna().all()
    assert source.legal_status.loc[~suppressed, "employment_thousand"].notna().all()
    assert set(source.size_bands["sector"]) == {
        "private_sector",
        "whole_economy",
        "central_and_local_government",
        "non_profit",
    }
    # Table 1 and Table 2 publish the same private sector.
    t1 = source.size_bands[source.size_bands["table"] == "Table 1"]
    t2 = source.size_bands[
        (source.size_bands["table"] == "Table 2")
        & (source.size_bands["sector"] == "private_sector")
    ]
    assert (
        t1.loc[t1["size_band"] == "All businesses", "business_number"].iloc[0]
        == t2.loc[t2["size_band"] == "All", "business_number"].iloc[0]
    )


def test_reader_rejects_value_with_suppressed_flag(tmp_path) -> None:
    for name in (UK_FIRM_BPE_LEGAL_STATUS_FILE, UK_FIRM_BPE_SIZE_BANDS_FILE):
        (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())
    path = tmp_path / UK_FIRM_BPE_LEGAL_STATUS_FILE
    text = path.read_text(encoding="utf-8").replace(",,[c]", ",1,[c]", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="empty exactly"):
        read_uk_firm_bpe_source(tmp_path)


def test_tool_flags_suppressed_cells_and_headers_agree() -> None:
    tool = _load_tool()
    assert tool.LEGAL_STATUS_HEADER == UK_FIRM_BPE_LEGAL_STATUS_COLUMNS
    assert tool.SIZE_BANDS_HEADER == UK_FIRM_BPE_SIZE_BANDS_COLUMNS
    assert tool._cell("[c]", "x") == ("", "[c]")
    assert tool._cell(1749, "x") == ("1749", "")
    assert tool._band("With no employees (registered) [Note 2] [Note 6]") == (
        "With no employees (registered)"
    )
    with pytest.raises(ValueError):
        tool._cell("n/a", "x")


def test_bpe_employment_already_counts_working_proprietors() -> None:
    """Single-value rows' means sit on the proprietor rule's employment."""

    fits = uk_firm_small_headcount_fits(_source())
    single = fits[(fits["support_low"] == fits["support_high"]) & (fits.businesses > 0)]
    assert len(single) >= 4
    assert np.allclose(single["mean_employment"], single["support_low"], atol=0.06)


# -- proprietors, employees and bands ------------------------------------------


def test_proprietor_rules() -> None:
    status = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 6, 7])
    employment = np.array([1, 5, 1, 3, 1, 2, 7, 3, 3, 3, 3])
    assert uk_firm_assumed_proprietors(status, employment).tolist() == [
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        0,
        0,
        0,
        0,
    ]
    employees = uk_firm_employees(status, employment)
    assert employees.dtype == np.int64
    assert employees.tolist() == [1, 5, 0, 2, 0, 0, 5, 3, 3, 3, 3]
    assert (employees >= 0).all()
    with pytest.raises(ValueError):
        uk_firm_employees(np.array([8]), np.array([1]))
    with pytest.raises(ValueError):
        uk_firm_employees(np.array([1]), np.array([1.5]))


def test_employee_size_bands() -> None:
    lowers = [
        int(re.match(r"\d+", band).group()) for band in UK_FIRM_EMPLOYEE_SIZE_BANDS
    ]
    labels = uk_firm_employee_size_bands(np.array(lowers))
    assert labels.tolist() == list(UK_FIRM_EMPLOYEE_SIZE_BANDS)
    uppers = [edge - 1 for edge in lowers[1:]]
    assert uk_firm_employee_size_bands(np.array(uppers)).tolist() == list(
        UK_FIRM_EMPLOYEE_SIZE_BANDS[:-1]
    )
    assert uk_firm_employee_size_bands(np.array([123456])).tolist() == ["1000+"]
    assert len(UK_FIRM_EMPLOYEE_SIZE_BANDS) == 12
    assert UK_FIRM_EMPLOYEE_SIZE_BANDS[:3] == ("0", "1", "2-4")


# -- the target ----------------------------------------------------------------


def test_geometric_fit_matches_mean_and_clips() -> None:
    for low, high, mean in ((2, 4, 2.3), (3, 5, 4.8), (1, 4, 2.5), (4, 6, 4.01)):
        ratio, fitted, clipped = fit_truncated_geometric(low, high, mean)
        assert not clipped
        assert fitted == pytest.approx(mean, abs=1e-9)
        assert ratio > 0
    _, fitted, clipped = fit_truncated_geometric(2, 4, 4.5)
    assert clipped and fitted == pytest.approx(4, abs=1e-5)
    _, fitted, clipped = fit_truncated_geometric(2, 4, 1.0)
    assert clipped and fitted == pytest.approx(2, abs=1e-5)


def test_fits_match_published_means() -> None:
    fits = uk_firm_small_headcount_fits(_source())
    multi = fits[fits["support_high"] > fits["support_low"]]
    assert len(multi) == 3
    assert not multi["clipped"].any()
    assert np.allclose(multi["fitted_mean"], multi["mean_employment"], atol=1e-8)
    for _, row in multi.iterrows():
        section = UK_FIRM_SMALL_HEADCOUNT_STATUSES[row["ons_legal_status"]]
        businesses, employment = _table_3(section, row["bpe_size_band"])
        assert row["mean_employment"] == pytest.approx(employment / businesses)
    # Companies' "2 to 4" stays in band; sole and partnership rows spill over.
    company = multi[multi["ons_legal_status"] == 1].iloc[0]
    assert company["in_band_share"] == pytest.approx(1.0)
    assert (multi.loc[multi["ons_legal_status"] != 1, "in_band_share"] < 1).all()


def test_targets_follow_table_3() -> None:
    targets = uk_firm_small_headcount_targets(_source())
    assert np.allclose(targets.groupby("ons_legal_status")["share"].sum(), 1.0)
    by = targets.set_index(["ons_legal_status", "employment"])["businesses"]
    registered = "With no employees (registered)"
    assert by[(1, 1)] == _table_3("companies", registered)[0]
    assert by[(1, 2)] + by[(1, 3)] + by[(1, 4)] == pytest.approx(
        _table_3("companies", "2 to 4")[0]
    )
    assert by[(2, 1)] == _table_3("sole_proprietorships", registered)[0]
    assert by[(2, 2)] == _table_3("sole_proprietorships", "1")[0]
    assert by[(3, 1)] == 0
    assert by[(3, 2)] == _table_3("ordinary_partnerships", registered)[0]
    assert by[(3, 3)] == _table_3("ordinary_partnerships", "1")[0]
    # Geometric shape decreasing for companies (mean below the midpoint).
    assert by[(1, 2)] > by[(1, 3)] > by[(1, 4)]


def test_module_hard_codes_no_bpe_number() -> None:
    text = MODULE.read_text(encoding="utf-8")
    literals = {
        int(m)
        for m in re.findall(r"(?<![\w.])\d[\d_]*(?![\w.])", text.replace("_", ""))
    }
    source = _source()
    published = set()
    for frame, columns in (
        (source.legal_status, ("business_number", "employment_thousand")),
        (source.size_bands, UK_FIRM_BPE_SIZE_BANDS_COLUMNS[3:]),
    ):
        for column in columns:
            published |= {int(v) for v in frame[column].dropna() if v >= 100}
    # Band edges in labels are not statistics.
    edges = {
        int(e) for band in UK_FIRM_EMPLOYEE_SIZE_BANDS for e in re.findall(r"\d+", band)
    }
    assert (literals & published) - edges == set()


# -- the redraw ----------------------------------------------------------------


def test_redraw_touches_only_small_status_1_to_3_rows() -> None:
    before = _frame(20_000)
    after = redraw_uk_firm_small_headcount(before, _source(), seed=7)
    low, high = UK_FIRM_ONS_SMALL_BAND
    eligible = before["ons_legal_status"].isin([1, 2, 3]) & before[
        "employment"
    ].between(low, high)
    pd.testing.assert_frame_equal(
        after.loc[~eligible], before.loc[~eligible], check_exact=True
    )
    pd.testing.assert_frame_equal(
        after.drop(columns="employment"), before.drop(columns="employment")
    )
    assert after["employment"].dtype == before["employment"].dtype
    assert after.loc[eligible, "employment"].between(low, high).all()
    assert (
        after.loc[eligible, "employment"] != before.loc[eligible, "employment"]
    ).any()
    # ONS employment bands never change.
    edges = [0, 4, 9, 19, 49, 99, 249, np.inf]
    assert (
        pd.cut(after["employment"], edges).astype(str)
        == pd.cut(before["employment"], edges).astype(str)
    ).all()
    # Partnerships never redraw to employment 1.
    partnership = eligible & (before["ons_legal_status"] == 3)
    assert (after.loc[partnership, "employment"] != 1).all()
    assert UK_FIRM_HEADCOUNT_STATUS == "assumed"


def test_redraw_is_deterministic_and_row_order_free() -> None:
    frame = _frame(5_000, seed=3)
    one = redraw_uk_firm_small_headcount(frame, _source(), seed=11)
    two = redraw_uk_firm_small_headcount(frame, _source(), seed=11)
    pd.testing.assert_frame_equal(one, two)
    shuffled = frame.sample(frac=1.0, random_state=5)
    three = redraw_uk_firm_small_headcount(shuffled, _source(), seed=11)
    pd.testing.assert_frame_equal(three.loc[one.index], one)
    other = redraw_uk_firm_small_headcount(frame, _source(), seed=12)
    assert not other["employment"].equals(one["employment"])
    # Without id columns the index orders the draws.
    bare = frame.drop(columns=["firm_id", "source_firm_id"]).set_index(frame["firm_id"])
    four = redraw_uk_firm_small_headcount(bare, _source(), seed=11)
    assert four["employment"].tolist() == one["employment"].tolist()


def test_redraw_rejects_repeated_ids() -> None:
    frame = _frame(10)
    frame.loc[1, "source_firm_id"] = frame.loc[0, "source_firm_id"]
    with pytest.raises(ValueError, match="repeat"):
        redraw_uk_firm_small_headcount(frame, _source(), seed=1)


def test_weighted_shares_converge_to_targets() -> None:
    # Every row eligible: about 200,000 weighted draws per status.
    before = _frame(600_000, seed=9)
    rng = np.random.default_rng(1)
    before["ons_legal_status"] = rng.integers(1, 4, len(before))
    before["employment"] = rng.integers(1, 5, len(before))
    after = redraw_uk_firm_small_headcount(before, _source(), seed=2)
    table = uk_firm_small_headcount_diagnostics(before, after, _source())
    assert set(table["ons_legal_status"]) == {1, 2, 3}
    assert np.allclose(table["share_after"], table["target_share"], atol=0.006)
    # Before: the generator's even spread is far from the target.
    assert not np.allclose(table["share_before"], table["target_share"], atol=0.05)
    assert np.allclose(
        table.groupby("ons_legal_status")["weight_before"].sum(),
        table.groupby("ons_legal_status")["weight_after"].sum(),
        rtol=1e-6,
    )


# -- reconciliation ------------------------------------------------------------


def test_size_band_reconciliation() -> None:
    frame = _frame(20_000, seed=4)
    table = uk_firm_bpe_size_band_reconciliation(frame, _source(), "firm_weight")
    assert set(table["scope"]) == {"private_sector", "whole_economy"}
    weight = frame["firm_weight"].to_numpy(np.float64)
    whole = table[table["scope"] == "whole_economy"].set_index("size_band")
    assert whole.loc["All", "modelled_businesses"] == pytest.approx(weight.sum())
    assert whole.drop(index="All")["modelled_businesses"].sum() == pytest.approx(
        weight.sum()
    )
    employees = uk_firm_employees(frame["ons_legal_status"], frame["employment"])
    assert whole.loc["All", "modelled_employees"] == pytest.approx(weight @ employees)
    private = table[table["scope"] == "private_sector"].set_index("size_band")
    keep = frame["ons_legal_status"].isin([1, 2, 3, 4]).to_numpy()
    assert private.loc["All", "modelled_businesses"] == pytest.approx(
        weight[keep].sum()
    )
    # Unregistered businesses come out of the first band and the total only.
    gap = private["bpe_businesses"] - private["bpe_registered_businesses"]
    assert gap.loc["All"] == gap.loc["Zero or 1 employee"] > 0
    assert (gap.drop(index=["All", "Zero or 1 employee"]) == 0).all()
    assert list(private.index[:2]) == ["Zero or 1 employee", "2 to 4"]


def test_firm_headcount_exports() -> None:
    for name in firm_headcount.__all__:
        assert hasattr(firm_headcount, name)
