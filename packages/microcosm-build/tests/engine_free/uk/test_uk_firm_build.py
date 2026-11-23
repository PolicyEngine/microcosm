"""The local UK firm build writes a valid table and a reproducible receipt.

A thinned 2024-25 build on the paper-parity fixtures, with a test-only rules
double, runs twice into different directories: the parquet round-trips, the
receipt's hashes match the files, and the two builds agree exactly without a
committed golden hash. The tool is loaded from its path, refuses to run
without ``--rules`` and writes the same receipt as the library call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_build import (
    UK_FIRM_RECEIPT_FILE,
    UK_FIRM_TABLE_FILE,
    build_uk_firm_dataset,
)
from microcosm.build.uk_runtime.firm_employment_allowance import (
    UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS,
    UK_FIRM_EMPLOYMENT_ALLOWANCE_METHOD,
    uk_firm_employment_allowance_flags,
)
from microcosm.build.uk_runtime.firm_generation import (
    LARGE_EMPLOYMENT_CAP,
    LARGE_EMPLOYMENT_MEAN,
    ONS_EMPLOYMENT_BANDS,
    UKFirmGenerationConfig,
)
from microcosm.build.uk_runtime.firm_paybill import (
    UK_FIRM_PAYBILL_METHOD,
    UK_FIRM_PAYBILL_STATUS,
    UK_FIRM_PAYBILL_TAX_YEAR,
)
from microcosm.build.uk_runtime.firm_rules import (
    UKFirmRuleOutputs,
    rule_period_for_vintage,
)
from microcosm.build.uk_runtime.firm_table import (
    UK_FIRM_EXTENDED_TABLE_COLUMNS,
    UK_FIRM_PAYBILL_TABLE_COLUMNS,
    uk_firm_table_columns,
    validate_uk_firm_table,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURES = PATHS.tests / "fixtures" / "uk"
PROCESSED = FIXTURES / "firm_2024_25"
ATTRIBUTES = FIXTURES / "firm_attributes_2025"
PAYBILL = FIXTURES / "firm_paybill_2024"
SIZE = FIXTURES / "firm_2024_25_size"
TOOL = PATHS.repository / "tools" / "build_uk_firms.py"
SEED = 42
ITERATIONS = 20
WINDOW_FRACTION = 0.02
TAIL_FRACTION = 0.005


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


def _config(rules: _StandardRateDouble) -> UKFirmGenerationConfig:
    return UKFirmGenerationConfig(
        seed=SEED,
        n_iterations=ITERATIONS,
        sample_window_fraction=WINDOW_FRACTION,
        sample_tail_fraction=TAIL_FRACTION,
        firm_rules=rules,
    ).with_vintage("2024-25")


def _build(
    output_dir: Path,
    paybill_dir: Path | None = None,
    employment_allowance: bool = False,
    **sources: Path,
) -> dict:
    rules = _StandardRateDouble()
    return build_uk_firm_dataset(
        processed_dir=PROCESSED,
        attributes_dir=ATTRIBUTES,
        output_dir=output_dir,
        rules=rules,
        config=_config(rules),
        paybill_dir=paybill_dir,
        employment_allowance=employment_allowance,
        **sources,
    )


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    return (
        _build(tmp_path_factory.mktemp("first") / "out"),
        _build(tmp_path_factory.mktemp("second") / "out"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def _load_tool():
    # Registered before it runs, as the stock cell tool's test does.
    if "build_uk_firms" in sys.modules:
        return sys.modules["build_uk_firms"]
    spec = importlib.util.spec_from_file_location("build_uk_firms", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parquet_round_trips_and_validates(builds) -> None:
    built, _ = builds
    table_path = built["table_path"]

    assert table_path.name == UK_FIRM_TABLE_FILE
    written = pd.read_parquet(table_path)
    pd.testing.assert_frame_equal(written, built["table"])
    validate_uk_firm_table(written)
    receipt = built["receipt"]
    assert receipt["output"]["rows"] == len(written)
    assert receipt["output"]["columns"] == list(written.columns)
    assert receipt["output"]["weighted_firms"] == pytest.approx(
        float(written["firm_weight"].sum())
    )


def test_receipt_hashes_match_the_files(builds) -> None:
    built, _ = builds
    receipt = json.loads(built["receipt_path"].read_text(encoding="utf-8"))

    assert built["receipt_path"].name == UK_FIRM_RECEIPT_FILE
    assert receipt == built["receipt"]
    assert receipt["output"]["sha256"] == _sha256(built["table_path"])
    for role, directory in (("processed", PROCESSED), ("attributes", ATTRIBUTES)):
        expected = {path.name: _sha256(path) for path in directory.glob("*.csv")}
        assert expected
        assert receipt["inputs"][role] == expected


def test_receipt_records_the_run(builds) -> None:
    built, _ = builds
    receipt = built["receipt"]

    assert receipt["schema_version"] == 1
    assert receipt["data_vintage"] == "2024-25"
    assert receipt["seed"] == SEED
    assert receipt["rule_period"] == rule_period_for_vintage("2024-25")
    assert receipt["config"]["n_iterations"] == ITERATIONS
    assert "firm_rules" not in receipt["config"]
    assert "vat_rule_evaluator" not in receipt["config"]
    assert receipt["rules_provenance"] == [{"rules": "standard-rate test double"}]
    assert 0.0 < receipt["validation"]["overall"] <= 1.0
    assert receipt["target_diagnostics"]
    attributes = {
        row["attribute"] for row in receipt["uk_firm_attribute_fallback_counts"]
    }
    assert attributes == {"ons_region_code", "ons_legal_status"}
    # Without a paybill source nothing paybill-shaped is written.
    assert receipt["output"]["columns"] == list(UK_FIRM_EXTENDED_TABLE_COLUMNS)
    assert "paybill" not in receipt["inputs"]
    assert "weighted_paybill_gbp" not in receipt["output"]
    assert "uk_firm_paybill_fallback_counts" not in receipt
    assert "uk_firm_paybill" not in receipt
    assert "uk_firm_paybill_in_frame" not in receipt


def test_paybill_build_sets_firm_paybill_and_records_its_source(
    builds, tmp_path: Path
) -> None:
    built = _build(tmp_path / "paybill", paybill_dir=PAYBILL)
    table = built["table"]
    receipt = json.loads(built["receipt_path"].read_text(encoding="utf-8"))
    base = builds[0]["table"]

    assert tuple(table.columns) == UK_FIRM_PAYBILL_TABLE_COLUMNS
    validate_uk_firm_table(pd.read_parquet(built["table_path"]))
    # The paybill adds a column and changes nothing else.
    pd.testing.assert_frame_equal(table[list(UK_FIRM_EXTENDED_TABLE_COLUMNS)], base)
    assert receipt["inputs"]["paybill"] == {
        "ashe_section_annual_pay.csv": _sha256(PAYBILL / "ashe_section_annual_pay.csv")
    }
    weighted = float((table["firm_weight"] * table["firm_paybill"]).sum())
    assert receipt["output"]["weighted_paybill_gbp"] == pytest.approx(weighted)
    levels = {row["level"]: row for row in receipt["uk_firm_paybill_fallback_counts"]}
    assert set(levels) == {"section", "all_employees"}
    assert sum(row["firms"] for row in levels.values()) == len(table)
    assert sum(row["weighted_paybill_gbp"] for row in levels.values()) == pytest.approx(
        weighted
    )
    assert receipt["uk_firm_paybill"] == {
        "status": UK_FIRM_PAYBILL_STATUS,
        "method": UK_FIRM_PAYBILL_METHOD,
        "tax_year": UK_FIRM_PAYBILL_TAX_YEAR,
    }
    assert receipt["uk_firm_paybill"]["status"] == "assumed"
    split = {row["in_frame"]: row for row in receipt["uk_firm_paybill_in_frame"]}
    assert set(split) == {True, False}
    for flag, row in split.items():
        rows = table[table["in_frame"] == flag]
        assert row["weighted_paybill_gbp"] == pytest.approx(
            float((rows["firm_weight"] * rows["firm_paybill"]).sum())
        )
        assert row["weighted_employees"] == pytest.approx(
            float((rows["firm_weight"] * rows["firm_employees"]).sum())
        )
    assert sum(row["weighted_paybill_gbp"] for row in split.values()) == pytest.approx(
        weighted
    )
    assert str(PAYBILL) not in built["receipt_path"].read_text(encoding="utf-8")


def _dbt_large_firms() -> pd.DataFrame:
    return pd.read_csv(SIZE / "dbt_bpe_private_250plus.csv")


def test_dbt_large_firm_fixture_matches_its_provenance_and_the_mean() -> None:
    provenance = json.loads((SIZE / ".provenance.json").read_text(encoding="utf-8"))
    record = provenance["files"]["dbt_bpe_private_250plus.csv"]
    assert _sha256(SIZE / "dbt_bpe_private_250plus.csv") == record["sha256"]
    assert provenance["source"]["access_date"] == "2026-11-22"
    assert len(provenance["source"]["sha256"]) == 64
    large = _dbt_large_firms()
    assert int(large["businesses"].sum()) == 8_250
    mean = (
        1_000.0
        * float(large["employment_thousand"].sum())
        / float(large["businesses"].sum())
    )
    assert LARGE_EMPLOYMENT_MEAN == pytest.approx(mean)


def test_large_firm_employment_matches_the_dbt_mean_without_a_clamp(
    builds, tmp_path: Path
) -> None:
    """With the DBT mean set, the in-frame 250+ band's weighted mean
    headcount is DBT's within 15% and no clamp at 2,000 remains. Out-of-frame
    rows draw below 250, which can move their legal status (drawn jointly
    with the employment band); every other column, and every headcount the
    default build draws below 250, matches the default build."""

    rules = _StandardRateDouble()
    built = build_uk_firm_dataset(
        processed_dir=PROCESSED,
        attributes_dir=ATTRIBUTES,
        output_dir=tmp_path / "out",
        rules=rules,
        config=dataclasses.replace(
            _config(rules), large_employment_mean=LARGE_EMPLOYMENT_MEAN
        ),
    )
    table = built["table"]
    base = builds[0]["table"]
    in_frame = base["in_frame"].to_numpy(bool)
    np.testing.assert_array_equal(table["in_frame"].to_numpy(bool), in_frame)
    band_dependent = [
        "firm_employees",
        "firm_legal_form",
        "firm_is_public_body",
        "ons_legal_status",
    ]
    pd.testing.assert_frame_equal(
        table.drop(columns=band_dependent), base.drop(columns=band_dependent)
    )
    pd.testing.assert_frame_equal(
        table.loc[in_frame, band_dependent[1:]], base.loc[in_frame, band_dependent[1:]]
    )
    legacy = base["firm_employees"].to_numpy(dtype=np.float64)
    assert legacy.max() <= 2_000
    employees = table["firm_employees"].to_numpy(dtype=np.float64)
    weights = table["firm_weight"].to_numpy(dtype=np.float64)
    large = employees >= 250
    np.testing.assert_array_equal(large, (legacy >= 250) & in_frame)
    assert not large[~in_frame].any()
    below = legacy < 250
    np.testing.assert_array_equal(employees[below], legacy[below])
    weighted_mean = float((weights * employees)[large].sum() / weights[large].sum())
    assert weighted_mean == pytest.approx(LARGE_EMPLOYMENT_MEAN, rel=0.15)
    # The default clamps the band to [250, 2000]; no clamp at 2000 remains.
    assert ONS_EMPLOYMENT_BANDS["250+"][1] == 2_000 < LARGE_EMPLOYMENT_CAP
    assert employees.max() > 2_000
    assert employees.max() <= LARGE_EMPLOYMENT_CAP
    assert float((employees == 2_000).sum()) < 0.01 * float(large.sum())


def test_paybill_build_refuses_another_vintage(tmp_path: Path) -> None:
    """The ASHE 2025 table is tax year 2024-25; another vintage is refused
    before anything is generated or written."""

    rules = _StandardRateDouble()
    config = _config(rules).with_vintage("2023-24")
    with pytest.raises(ValueError, match="tax year 2024-25"):
        build_uk_firm_dataset(
            processed_dir=PROCESSED,
            attributes_dir=ATTRIBUTES,
            output_dir=tmp_path / "out",
            rules=rules,
            config=config,
            paybill_dir=PAYBILL,
        )
    assert not (tmp_path / "out").exists()


def _comparable(receipt: dict) -> dict:
    """A receipt without the parquet hash, floats rounded to 9 significant digits.

    Same-seed builds agree exactly on the machines tested so far; the
    comparison tolerates last-digit float differences so a platform's
    floating-point reductions cannot fail it.
    """

    def _round(value):
        if isinstance(value, float):
            return float(f"{value:.9g}")
        if isinstance(value, dict):
            return {key: _round(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_round(item) for item in value]
        return value

    trimmed = json.loads(json.dumps(receipt))
    trimmed["output"].pop("sha256")
    return _round(trimmed)


def test_two_builds_agree(builds) -> None:
    first, second = builds

    pd.testing.assert_frame_equal(first["table"], second["table"])
    assert _comparable(first["receipt"]) == _comparable(second["receipt"])


def test_receipt_holds_no_absolute_path(builds, tmp_path_factory) -> None:
    for built in builds:
        text = built["receipt_path"].read_text(encoding="utf-8")
        assert str(tmp_path_factory.getbasetemp()) not in text
        assert str(built["table_path"].parent) not in text
        assert str(PATHS.repository) not in text


def test_tool_needs_rules(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit) as raised:
        tool.main(
            [
                "--processed-dir",
                str(PROCESSED),
                "--attributes-dir",
                str(ATTRIBUTES),
                "--out-dir",
                str(tmp_path),
            ]
        )
    assert raised.value.code != 0
    assert not any(tmp_path.iterdir())


def test_tool_needs_an_artifact_for_axiom(tmp_path: Path) -> None:
    tool = _load_tool()
    with pytest.raises(SystemExit) as raised:
        tool.main(
            [
                "--processed-dir",
                str(PROCESSED),
                "--attributes-dir",
                str(ATTRIBUTES),
                "--out-dir",
                str(tmp_path),
                "--rules",
                "axiom",
            ]
        )
    assert raised.value.code != 0


def test_tool_writes_the_library_build(
    builds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The tool's flags reach the build: thinned to the test's fractions
    after ``--fast`` is seen, it writes the library build's receipt."""

    tool = _load_tool()
    real_config = tool._config
    seen: list[UKFirmGenerationConfig] = []

    def thinned_config(args, rules):
        config = real_config(args, rules)
        seen.append(config)
        return dataclasses.replace(
            config,
            sample_window_fraction=WINDOW_FRACTION,
            sample_tail_fraction=TAIL_FRACTION,
        )

    monkeypatch.setattr(tool, "_rules", lambda args: _StandardRateDouble())
    monkeypatch.setattr(tool, "_config", thinned_config)
    out = tmp_path / "tool"
    code = tool.main(
        [
            "--processed-dir",
            str(PROCESSED),
            "--attributes-dir",
            str(ATTRIBUTES),
            "--out-dir",
            str(out),
            "--rules",
            "policyengine-uk",
            "--fast",
            "--seed",
            str(SEED),
            "--n-iterations",
            str(ITERATIONS),
        ]
    )

    assert code == 0
    assert str(out / UK_FIRM_TABLE_FILE) in capsys.readouterr().out
    (config,) = seen
    fast = UKFirmGenerationConfig.fast()
    assert config.sample_window_fraction == fast.sample_window_fraction
    assert config.sample_tail_fraction == fast.sample_tail_fraction
    assert config.n_iterations == ITERATIONS
    assert config.data_vintage == "2024-25"
    receipt = json.loads((out / UK_FIRM_RECEIPT_FILE).read_text(encoding="utf-8"))
    assert _comparable(receipt) == _comparable(builds[0]["receipt"])


def test_employment_allowance_build_adds_only_the_assumed_flags(
    builds, tmp_path: Path
) -> None:
    built = _build(tmp_path / "allowance", employment_allowance=True)
    table = built["table"]
    receipt = json.loads(built["receipt_path"].read_text(encoding="utf-8"))
    base = builds[0]["table"]

    assert tuple(table.columns) == uk_firm_table_columns(
        ["attributes", "employment_allowance"]
    )
    validate_uk_firm_table(pd.read_parquet(built["table_path"]))
    # The flags read no random numbers: every other column is unchanged.
    pd.testing.assert_frame_equal(table[list(UK_FIRM_EXTENDED_TABLE_COLUMNS)], base)
    flags = uk_firm_employment_allowance_flags(
        table["ons_legal_status"].to_numpy(), table["firm_employees"].to_numpy()
    )
    for column, values in flags.items():
        assert table[column].to_numpy(bool).tolist() == values.tolist()
    record = receipt["uk_firm_employment_allowance"]
    assert record["status"] == "assumed"
    assert record["method"] == UK_FIRM_EMPLOYMENT_ALLOWANCE_METHOD
    weight = table["firm_weight"].to_numpy(float)
    for column in UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS:
        assert record["weighted_firms_where_true"][column] == pytest.approx(
            float(weight[table[column].to_numpy(bool)].sum())
        )
    assert "uk_firm_employment_allowance" not in builds[0]["receipt"]


def test_full_opt_in_build_adds_the_assumed_inputs(builds, tmp_path: Path) -> None:
    """Every opt-in source at once: public sector headcounts change only
    central government and local authority rows, and profit and rateable
    value add one column each."""

    rules = _StandardRateDouble()
    built = build_uk_firm_dataset(
        processed_dir=PROCESSED,
        attributes_dir=ATTRIBUTES,
        output_dir=tmp_path / "out",
        rules=rules,
        config=_config(rules),
        paybill_dir=PAYBILL,
        employment_allowance=True,
        public_sector_dir=FIXTURES / "firm_public_sector_2025",
        profit_dir=FIXTURES / "firm_ct_2024",
        rateable_value_dir=FIXTURES / "firm_rateable_value_2024",
        pay_distribution_dir=FIXTURES / "firm_pay_distribution_2024",
    )
    table = built["table"]
    receipt = built["receipt"]
    base = builds[0]["table"]
    assert tuple(table.columns) == uk_firm_table_columns(
        [
            "attributes",
            "paybill",
            "employment_allowance",
            "pay_distribution",
            "profit",
            "rateable_value",
        ]
    )
    validate_uk_firm_table(pd.read_parquet(built["table_path"]))
    public = base["ons_legal_status"].isin([5, 6]).to_numpy()
    pd.testing.assert_frame_equal(
        table.loc[~public, list(base.columns)], base.loc[~public]
    )
    pd.testing.assert_frame_equal(
        table.loc[public, list(base.columns)].drop(columns="firm_employees"),
        base.loc[public].drop(columns="firm_employees"),
    )
    weight = table["firm_weight"].to_numpy(float)
    for status, sector in ((5, "central_government"), (6, "local_government")):
        rows = table["ons_legal_status"].to_numpy() == status
        record = next(
            row
            for row in receipt["uk_firm_public_sector"]["sectors"].values()
            if row["ons_legal_status"] == status
        )
        assert float((weight * table["firm_employees"])[rows].sum()) == (
            pytest.approx(record["target_weighted_employment"], rel=1e-3)
        ), sector
    profit = table["firm_taxable_trading_profit"].to_numpy(float)
    assert profit.max() > 0
    assert not profit[table["ons_legal_status"].to_numpy() != 1].any()
    assert receipt["uk_firm_profit"]["status"] == "assumed"
    assert receipt["uk_firm_profit"]["weighted_profit_gbp"] == pytest.approx(
        float((weight * profit).sum())
    )
    value = table["firm_rateable_value"].to_numpy(float)
    assert value.max() > 0
    assert not value[~table["firm_in_england"].to_numpy(bool)].any()
    assert receipt["uk_firm_rateable_value"]["status"] == "assumed"
    below = table["firm_employees_below_secondary_threshold"].to_numpy()
    assert 0 < below.sum() and (below <= table["firm_employees"].to_numpy()).all()
    assert receipt["uk_firm_pay_distribution"]["threshold_gbp"] == 9_100.0
    for role in ("public_sector", "profit", "rateable_value", "pay_distribution"):
        assert len(receipt["inputs"][role]) == 1
    for key in ("uk_firm_public_sector", "uk_firm_profit", "uk_firm_rateable_value"):
        assert key not in builds[0]["receipt"]


def test_labour_link_build_sets_employees_and_draws_allowance_takeup(
    builds, tmp_path: Path
) -> None:
    """BPE redraws only 0-4 band headcounts of statuses 1-3; firm_employees
    drops the assumed working proprietors; the take-up draw stays within the
    rule flag."""

    rules = _StandardRateDouble()
    built = build_uk_firm_dataset(
        processed_dir=PROCESSED,
        attributes_dir=ATTRIBUTES,
        output_dir=tmp_path / "out",
        rules=rules,
        config=_config(rules),
        paybill_dir=PAYBILL,
        employment_allowance=True,
        bpe_dir=FIXTURES / "firm_bpe_2025",
        employment_allowance_takeup_dir=FIXTURES
        / "firm_employment_allowance_takeup_2024_25",
    )
    table = built["table"]
    receipt = built["receipt"]
    base = builds[0]["table"]
    assert tuple(table.columns) == uk_firm_table_columns(
        [
            "attributes",
            "paybill",
            "employment_allowance",
            "sic_division",
            "labour_link",
            "employment_allowance_takeup",
        ]
    )
    validate_uk_firm_table(pd.read_parquet(built["table_path"]))
    pd.testing.assert_series_equal(table["firm_id"], base["firm_id"])
    before = base["firm_employees"].to_numpy()
    after = table["ons_employment"].to_numpy()
    status = table["ons_legal_status"].to_numpy()
    changed = before != after
    assert changed.any()
    small = (before >= 1) & (before <= 4) & np.isin(status, [1, 2, 3])
    assert not changed[~small].any()
    assert ((after[small] >= 1) & (after[small] <= 4)).all()
    employees = table["firm_employees"].to_numpy()
    assert (employees <= after).all()
    assert (employees[status == 1] == after[status == 1]).all()
    assert (employees[status == 2] == after[status == 2] - 1).all()
    claims = table["firm_claims_employment_allowance"].to_numpy(bool)
    rule = table["firm_employment_allowance_rule_claims"].to_numpy(bool)
    assert not (claims & ~rule).any()
    assert receipt["uk_firm_headcount"]["status"] == "assumed"
    assert receipt["uk_firm_employment_allowance_takeup"]["status"] == "assumed"
    for key in ("uk_firm_headcount", "uk_firm_employment_allowance_takeup"):
        assert key not in builds[0]["receipt"]


def test_consumption_build_splits_turnover_only(builds, tmp_path: Path) -> None:
    built = _build(
        tmp_path / "consumption", consumption_dir=FIXTURES / "firm_supply_use_2024"
    )
    table = built["table"]
    base = builds[0]["table"]
    assert tuple(table.columns) == uk_firm_table_columns(
        ["attributes", "sic_division", "consumption"]
    )
    validate_uk_firm_table(pd.read_parquet(built["table_path"]))
    pd.testing.assert_frame_equal(table[list(base.columns)], base)
    parts = table[
        ["firm_household_sales", "firm_intermediate_sales", "firm_other_final_sales"]
    ].sum(axis=1)
    np.testing.assert_allclose(parts, table["firm_turnover"], rtol=1e-9)
    assert (table["firm_household_sales"] > 0).any()
    record = built["receipt"]["uk_firm_consumption"]
    assert record["status"] == "assumed"
    assert all(
        abs(row["difference"]) <= 0.5 for row in record["table3_consumer_trends"]
    )
    shares = pd.read_csv(
        built["table_path"].parent / record["division_coicop_shares_file"],
        index_col="sic_division",
    )
    coicop = [f"{k:02d}" for k in range(1, 13)]
    np.testing.assert_allclose(shares[coicop].sum(axis=1), 1.0, rtol=1e-9)
    assert set(table["firm_sic_division"]) <= set(shares.index)
    assert "uk_firm_consumption" not in builds[0]["receipt"]


def test_labour_link_sources_need_their_prerequisites(tmp_path: Path) -> None:
    rules = _StandardRateDouble()
    with pytest.raises(ValueError, match="bpe_dir needs paybill_dir"):
        build_uk_firm_dataset(
            processed_dir=PROCESSED,
            attributes_dir=ATTRIBUTES,
            output_dir=tmp_path / "out",
            rules=rules,
            config=_config(rules),
            bpe_dir=FIXTURES / "firm_bpe_2025",
        )
    with pytest.raises(ValueError, match="needs bpe_dir and employment_allowance"):
        build_uk_firm_dataset(
            processed_dir=PROCESSED,
            attributes_dir=ATTRIBUTES,
            output_dir=tmp_path / "out",
            rules=rules,
            config=_config(rules),
            paybill_dir=PAYBILL,
            bpe_dir=FIXTURES / "firm_bpe_2025",
            employment_allowance_takeup_dir=FIXTURES
            / "firm_employment_allowance_takeup_2024_25",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("argument", "fixture"),
    [
        ("bpe_dir", "firm_bpe_2025"),
        ("consumption_dir", "firm_supply_use_2024"),
        ("employment_allowance_takeup_dir", "firm_employment_allowance_takeup_2024_25"),
        ("public_sector_dir", "firm_public_sector_2025"),
        ("profit_dir", "firm_ct_2024"),
        ("rateable_value_dir", "firm_rateable_value_2024"),
        ("pay_distribution_dir", "firm_pay_distribution_2024"),
    ],
)
def test_opt_in_sources_refuse_another_vintage(
    argument: str, fixture: str, tmp_path: Path
) -> None:
    rules = _StandardRateDouble()
    with pytest.raises(ValueError, match="tax year 2024-25"):
        build_uk_firm_dataset(
            processed_dir=PROCESSED,
            attributes_dir=ATTRIBUTES,
            output_dir=tmp_path / "out",
            rules=rules,
            config=_config(rules).with_vintage("2023-24"),
            **{argument: FIXTURES / fixture},
        )
    assert not (tmp_path / "out").exists()
