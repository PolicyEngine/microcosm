from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse

from microcosm.build.uk_runtime import (
    AxiomVATRuleEvaluator as PublicAxiomVATRuleEvaluator,
)
from microcosm.build.uk_runtime import firm_generation
from microcosm.build.uk_runtime.firm_generation import (
    DEFAULT_UK_FIRM_TARGET_PROFILE,
    FAST_SAMPLE_TAIL_FRACTION,
    FAST_SAMPLE_WINDOW_FRACTION,
    HMRC_BAND_COLUMNS,
    INPUT_FILES,
    INPUT_RATIO_LOWER,
    INPUT_RATIO_UPPER,
    LARGE_EMPLOYMENT_CAP,
    LARGE_EMPLOYMENT_MEAN,
    UK_FIRM_TARGET_IDS,
    VAT_LIABILITY_BANDS_CALIBRATED,
    AxiomVATRuleEvaluator,
    UKFirmGenerationConfig,
    assign_employment,
    build_firm_target_matrix,
    employment_band_index_array,
    employment_band_name,
    evaluate_vat_liability_if_registered_k,
    generate_base_firms,
    generate_input_values,
    generate_uk_firm_population,
    hmrc_band_name,
    large_employment_pareto_shape,
    large_employment_slice_means,
    read_uk_firm_source_data,
    resolve_uk_firm_rules,
    uk_firm_source_data_from_frames,
    uk_firm_source_data_from_ledger_facts,
    uk_firm_target_profile_from_mapping,
    write_uk_firm_population,
)
from microcosm.build.uk_runtime.firm_rules import (
    UK_FIRM_RULE_PERIOD_BY_VINTAGE,
    LegacyVATEvaluatorRules,
    UKFirmRuleOutputs,
)
from microcosm.build.uk_runtime.firm_table import (
    to_policyengine_uk_firm_table,
    validate_uk_firm_table,
)
from microcosm.build.uk_runtime.firm_universe import HMRC_BAND_NAMES


def test_axiom_vat_rule_evaluator_is_public_uk_runtime_api() -> None:
    assert PublicAxiomVATRuleEvaluator is AxiomVATRuleEvaluator


def test_uk_firm_source_data_reads_processed_directory(tmp_path: Path) -> None:
    frames = _source_frames()
    for key, filename in INPUT_FILES.items():
        frames[key].to_csv(tmp_path / filename, index=False)

    data = read_uk_firm_source_data(tmp_path)

    assert data.ons_total == 7
    assert data.hmrc_bands["Negative_or_Zero"] == 3.0
    assert data.vat_liability_bands["£Threshold_to_£150k"] == 1.2


def test_uk_firm_source_data_from_ledger_facts_uses_ledger_targets() -> None:
    data = uk_firm_source_data_from_ledger_facts(
        _ledger_facts(),
        data_vintage="2024-25",
    )

    assert data.ons_total == 7
    assert data.ons_turnover["SIC Code"].tolist() == [11, 12]
    assert data.ons_turnover.loc[0, "0-49"] == 2.0
    assert data.ons_turnover.loc[1, "500-999"] == 1.0
    assert data.ons_employment.loc[1, "20-49"] == 1.0
    assert data.hmrc_bands["Negative_or_Zero"] == 3.0
    assert data.hmrc_bands["£Threshold_to_£150k"] == 2.0
    assert data.vat_liability_bands["£Threshold_to_£150k"] == 1.2
    assert data.hmrc_population_sector["Trade_Sector"].tolist() == [11, 12]
    assert data.hmrc_population_sector["2024-25"].tolist() == [3.0, 2.0]
    assert data.hmrc_liability_sector["2024-25"].tolist() == [2.0, 1.5]


def test_uk_firm_source_data_from_ledger_facts_uses_target_profile_mapping() -> None:
    custom_record_set_id = "custom.ons.enterprise_count.by_sic_turnover_band"
    facts = _replace_fact_record_set(
        _ledger_facts(),
        old_record_set_id=DEFAULT_UK_FIRM_TARGET_PROFILE.ons_sic_turnover_record_set,
        new_record_set_id=custom_record_set_id,
    )
    target_profile = _ledger_target_profile_mapping(
        ons_sic_turnover=custom_record_set_id,
    )

    data = uk_firm_source_data_from_ledger_facts(
        facts,
        data_vintage="2024-25",
        target_profile=target_profile,
    )

    assert data.ons_turnover.loc[0, "0-49"] == 2.0
    assert data.ons_turnover.loc[1, "500-999"] == 1.0


def test_uk_firm_target_profile_from_mapping_requires_declared_targets() -> None:
    target_profile = _ledger_target_profile_mapping()
    missing_target_id = UK_FIRM_TARGET_IDS["hmrc_liability_sic"]
    target_profile["targets"] = [
        row
        for row in target_profile["targets"]
        if row["target_id"] != missing_target_id
    ]

    with pytest.raises(ValueError, match=missing_target_id):
        uk_firm_target_profile_from_mapping(target_profile)


def test_generate_uk_firm_population_accepts_ledger_source_data() -> None:
    data = uk_firm_source_data_from_ledger_facts(
        _ledger_facts(),
        data_vintage="2024-25",
    )

    result = generate_uk_firm_population(
        data,
        _firm_config(data_vintage="2024-25", n_iterations=4, seed=7),
    )

    diagnostics = result.target_diagnostics.set_index("target_name")
    # Ledger sector counts 3 and 2 are scaled onto the ten HMRC band traders,
    # as the paper allocates unknown-band traders proportionally.
    assert diagnostics.loc["hmrc_vat_population_sector/11", "target"] == 6.0
    assert diagnostics.loc["hmrc_vat_population_sector/12", "target"] == 4.0
    assert len(result.firms[result.firms["annual_turnover_k"] == 0]) == 3
    assert result.calibration.final_loss <= result.calibration.initial_loss


def test_uk_firm_source_data_from_ledger_facts_requires_complete_profile() -> None:
    facts = [
        fact
        for fact in _ledger_facts()
        if not (
            fact["layout"]["record_set_id"]
            == "hmrc.vat.fy2024_25.net_liability.by_turnover_band"
            and fact["layout"]["groupby_value_id"] == "500k_to_1m"
        )
    ]

    with pytest.raises(ValueError, match="£500k_to_£1m"):
        uk_firm_source_data_from_ledger_facts(facts, data_vintage="2024-25")


def test_generate_uk_firm_population_returns_experimental_firm_rows() -> None:
    data = _source_data()
    config = UKFirmGenerationConfig(
        vat_rule_evaluator=_TestVATRuleEvaluator(),
        data_vintage="2024-25",
        n_iterations=8,
        seed=7,
    )

    result = generate_uk_firm_population(data, config)
    firms = result.firms

    assert list(firms.columns) == [
        "firm_id",
        "source_firm_id",
        "source_firm_key",
        "data_vintage",
        "sic_code",
        "annual_turnover_k",
        "annual_input_k",
        "vat_liability_k",
        "employment",
        "firm_weight",
        "base_weight",
        "in_frame",
        "vat_scope",
        "vat_registered",
        "generation_status",
    ]
    assert firms["firm_id"].tolist() == sorted(firms["firm_id"].tolist())
    assert firms["firm_id"].is_unique
    assert firms["source_firm_key"].is_unique
    assert set(firms["generation_status"]) == {"experimental"}
    assert (firms["firm_weight"] > 0).all()
    # An unthinned build starts every row from a unit base weight.
    assert (firms["base_weight"] == 1.0).all()

    # The HMRC negative/zero-turnover traders are appended before calibration:
    # they are the only out-of-frame rows and are registered, in-scope traders.
    zero_turnover = firms[firms["annual_turnover_k"] == 0]
    assert len(zero_turnover) == 3
    assert firms.index[~firms["in_frame"]].tolist() == zero_turnover.index.tolist()
    assert zero_turnover["vat_registered"].all()
    assert zero_turnover["vat_scope"].all()
    assert len(result.calibration.weights) == len(firms)
    np.testing.assert_allclose(
        result.calibration.weights.numpy(),
        firms["firm_weight"].to_numpy(),
    )

    # Registration implies scope; every below-threshold frame firm is in scope.
    assert not (firms["vat_registered"] & ~firms["vat_scope"]).any()
    below = (
        firms["in_frame"]
        & (firms["annual_turnover_k"] > 0)
        & (firms["annual_turnover_k"] <= config.vat_threshold)
    )
    assert firms.loc[below, "vat_scope"].all()
    # Unregistered firms owe no VAT in the output.
    assert (firms.loc[~firms["vat_registered"], "vat_liability_k"] == 0).all()

    assert 0.0 <= result.validation.overall <= 1.0
    assert result.calibration.final_loss <= result.calibration.initial_loss
    assert len(result.calibration.loss_trajectory) == config.n_iterations
    assert (
        result.target_diagnostics["target_name"].str.startswith(("hmrc_", "ons_")).all()
    )
    assert "initial_estimate" in result.target_diagnostics.columns

    # ONS-frame rows count frame firms only; the appended traders are outside.
    frame = firms[firms["in_frame"]].copy()
    frame["employment_band"] = frame["employment"].apply(employment_band_name)
    weighted_employment = frame.groupby("employment_band")["firm_weight"].sum()
    diagnostics = result.target_diagnostics.set_index("target_name")
    for band in ("0-4", "5-9", "10-19", "20-49"):
        assert diagnostics.loc[
            f"ons_firm_employment/{band}",
            "estimate",
        ] == pytest.approx(float(weighted_employment.get(band, 0.0)), rel=1e-5)
    assert diagnostics.loc["ons_firm_population/total", "target"] == 7.0
    assert diagnostics.loc["ons_firm_population/total", "estimate"] == pytest.approx(
        float(frame["firm_weight"].sum()), rel=1e-5
    )
    assert diagnostics.loc[
        "ons_firm_population/total", "initial_estimate"
    ] == pytest.approx(7.0)
    # The appended traders enter the HMRC Negative_or_Zero row with propensity 1.
    assert diagnostics.loc["hmrc_vat_population/Negative_or_Zero", "target"] == 3.0
    assert diagnostics.loc[
        "hmrc_vat_population/Negative_or_Zero", "estimate"
    ] == pytest.approx(float(zero_turnover["firm_weight"].sum()), rel=1e-5)


def test_generate_uk_firm_population_uses_configured_vintage_targets() -> None:
    data = _source_data()

    result_2023 = generate_uk_firm_population(
        data,
        _firm_config(data_vintage="2023-24", n_iterations=2, seed=7),
    )
    result_2024 = generate_uk_firm_population(
        data,
        _firm_config(data_vintage="2024-25", n_iterations=2, seed=7),
    )

    targets_2023 = result_2023.target_diagnostics.set_index("target_name")["target"]
    targets_2024 = result_2024.target_diagnostics.set_index("target_name")["target"]
    assert targets_2023["hmrc_vat_population/£Threshold_to_£150k"] == 4.0
    assert targets_2024["hmrc_vat_population/£Threshold_to_£150k"] == 2.0
    # Sector counts are scaled to each vintage's HMRC band total.
    frames = _source_frames()
    for vintage, targets in (("2023-24", targets_2023), ("2024-25", targets_2024)):
        bands = frames["hmrc_population_band"].set_index("Financial_Year")
        band_total = float(bands.loc[vintage, list(HMRC_BAND_COLUMNS)].sum())
        sectors = frames["hmrc_population_sector"]
        sectors = sectors[sectors["Trade_Sector"] != "Total"]
        scale = band_total / float(sectors[vintage].sum())
        assert targets["hmrc_vat_population_sector/11"] == pytest.approx(
            float(sectors.loc[sectors["Trade_Sector"] == 11, vintage].iloc[0]) * scale
        )
    assert targets_2023["hmrc_vat_population_sector/11"] == pytest.approx(11 / 3)
    assert targets_2024["hmrc_vat_population_sector/11"] == pytest.approx(6.0)


def test_generate_uk_firm_population_requires_requested_hmrc_vintage() -> None:
    frames = _source_frames()
    frames["hmrc_population_band"] = frames["hmrc_population_band"].iloc[[1]]
    data = uk_firm_source_data_from_frames(**frames)

    with pytest.raises(ValueError, match="does not include vintage '2023-24'"):
        generate_uk_firm_population(
            data,
            _firm_config(data_vintage="2023-24", n_iterations=1),
        )


def test_generate_uk_firm_population_is_seed_reproducible() -> None:
    data = _source_data()
    config = UKFirmGenerationConfig(
        vat_rule_evaluator=_TestVATRuleEvaluator(),
        n_iterations=3,
        seed=11,
    )

    first = generate_uk_firm_population(data, config).firms
    second = generate_uk_firm_population(data, config).firms

    pd.testing.assert_frame_equal(first, second)


def test_write_uk_firm_population_writes_csv(tmp_path: Path) -> None:
    result = generate_uk_firm_population(
        _source_data(),
        _firm_config(n_iterations=2, seed=1),
    )

    path = write_uk_firm_population(result, tmp_path / "firms.csv")

    reloaded = pd.read_csv(path)
    assert reloaded["firm_id"].tolist() == result.firms["firm_id"].tolist()
    assert "firm_weight" in reloaded.columns


@pytest.mark.parametrize(
    ("turnover", "expected"),
    [
        (0.0, "Negative_or_Zero"),
        (85.0, "£1_to_Threshold"),
        (90.0, "£Threshold_to_£150k"),
        (300.0, "£150k_to_£300k"),
        (10_000.0, "£1m_to_£10m"),
        (10_001.0, "Greater_than_£10m"),
    ],
)
def test_hmrc_band_name_uses_configurable_threshold(
    turnover: float,
    expected: str,
) -> None:
    assert hmrc_band_name(turnover, threshold=85.0) == expected


@pytest.mark.parametrize(
    ("employment", "expected"),
    [(1, "0-4"), (9, "5-9"), (49, "20-49"), (250, "250+")],
)
def test_employment_band_name(employment: int, expected: str) -> None:
    assert employment_band_name(employment) == expected


def test_generate_uk_firm_population_requires_vat_rule_evaluator() -> None:
    with pytest.raises(ValueError, match="requires a VAT rule evaluator"):
        generate_uk_firm_population(
            _source_data(),
            UKFirmGenerationConfig(n_iterations=1, seed=1),
        )


def test_vat_liability_uses_input_tax_credit_not_full_input_cost() -> None:
    result = evaluate_vat_liability_if_registered_k(
        resolve_uk_firm_rules(_firm_config()),
        pd.DataFrame(
            {
                "firm_id": [0, 1],
                "sic_code": [11, 11],
                "annual_turnover_k": [100.0, 100.0],
                "annual_input_k": [40.0, 120.0],
                "employment": [1.0, 1.0],
            }
        ),
        "2023-24",
    )

    assert result.tolist() == pytest.approx([12.0, -4.0])


def test_firm_generation_uses_only_public_calibrate_names() -> None:
    """microcosm#636: no private ``microcosm.calibrate.solve`` imports."""

    tree = ast.parse(Path(firm_generation.__file__).read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("microcosm.calibrate")
        for alias in node.names
    ]
    assert "solve_constraint_system" in imported
    assert not [name for name in imported if name.startswith("_")]


def test_build_firm_target_matrix_is_float32_csr_with_two_universes() -> None:
    data = _source_data()
    config = _firm_config(data_vintage="2024-25")
    turnover = np.array([30.0, 120.0, 400.0, 0.0])
    sic = np.array([11, 12, 11, 12])
    employment_band = np.array([0, 1, 3, 0])
    frame = np.array([True, True, True, False])
    propensity = np.array([0.5, 0.25, 1.0, 1.0])
    liability_k = np.array([2.0, 4.0, 8.0, 0.0])

    matrix, targets, layout, names = build_firm_target_matrix(
        config,
        turnover,
        sic,
        employment_band,
        data,
        propensity=propensity,
        frame_mask=frame,
        vat_liability_if_registered_k=liability_k,
    )

    assert sparse.issparse(matrix)
    assert matrix.format == "csr"
    assert matrix.dtype == np.float32
    assert matrix.shape == (layout.n_targets, 4)
    assert len(names) == len(targets) == layout.n_targets
    dense = matrix.toarray()
    row = {name: dense[index] for index, name in enumerate(names)}
    # HMRC registered-trader rows carry the propensity.
    np.testing.assert_allclose(
        row["hmrc_vat_population/Negative_or_Zero"], [0, 0, 0, 1]
    )
    np.testing.assert_allclose(
        row["hmrc_vat_population/£1_to_Threshold"], [0.5, 0, 0, 0]
    )
    np.testing.assert_allclose(
        row["hmrc_vat_population/£Threshold_to_£150k"], [0, 0.25, 0, 0]
    )
    np.testing.assert_allclose(row["hmrc_vat_population_sector/11"], [0.5, 0, 1, 0])
    np.testing.assert_allclose(row["hmrc_vat_population_sector/12"], [0, 0.25, 0, 1])
    # ONS-frame rows exclude the appended trader.
    np.testing.assert_allclose(row["ons_firm_population/total"], [1, 1, 1, 0])
    np.testing.assert_allclose(row["ons_firm_employment/0-4"], [1, 0, 0, 0])
    np.testing.assert_allclose(row["ons_firm_employment/20-49"], [0, 0, 1, 0])
    # Net-VAT rows: liability-if-registered x propensity; the below-threshold
    # band is reported, not calibrated.
    assert "hmrc_vat_liability/£1_to_Threshold" not in row
    assert [name for name in names if name.startswith("hmrc_vat_liability/")] == [
        f"hmrc_vat_liability/{band}" for band in VAT_LIABILITY_BANDS_CALIBRATED
    ]
    np.testing.assert_allclose(
        row["hmrc_vat_liability/£Threshold_to_£150k"], [0, 1.0, 0, 0]
    )
    np.testing.assert_allclose(row["hmrc_vat_liability/£300k_to_£500k"], [0, 0, 8.0, 0])
    target = dict(zip(names, targets, strict=True))
    assert target["ons_firm_population/total"] == 7.0
    assert target["hmrc_vat_population/Negative_or_Zero"] == 3.0
    assert target["hmrc_vat_liability/£500k_to_£1m"] == pytest.approx(2_000.0)


def test_registered_trader_rows_are_propensity_scaled() -> None:
    result = generate_uk_firm_population(
        _source_data(),
        _firm_config(data_vintage="2024-25", n_iterations=2, seed=3),
    )
    firms = result.firms
    diagnostics = result.target_diagnostics.set_index("target_name")
    threshold = result.config.vat_threshold
    bands = firms["annual_turnover_k"].apply(lambda t: hmrc_band_name(t, threshold))
    # At unit base weights each band's expected registered count is its HMRC
    # count, capped by the frame mass the band holds.
    for band in HMRC_BAND_COLUMNS[1:]:
        frame_mass = float(firms.loc[firms["in_frame"] & (bands == band)].shape[0])
        target = diagnostics.loc[f"hmrc_vat_population/{band}", "target"]
        assert diagnostics.loc[
            f"hmrc_vat_population/{band}", "initial_estimate"
        ] == pytest.approx(min(target, frame_mass), rel=1e-6)


def test_vat_liability_rows_align_rule_outputs_by_firm_id() -> None:
    rules = _ShuffledFirmIdRules()
    result = generate_uk_firm_population(
        _source_data(),
        UKFirmGenerationConfig(
            data_vintage="2024-25",
            firm_rules=rules,
            n_iterations=2,
            seed=5,
        ),
    )
    firms = result.firms
    threshold = result.config.vat_threshold
    hmrc = _source_frames()["hmrc_population_band"].iloc[1]
    bands = firms["annual_turnover_k"].apply(lambda t: hmrc_band_name(t, threshold))
    diagnostics = result.target_diagnostics.set_index("target_name")
    assert rules.periods == [2024]
    for band in VAT_LIABILITY_BANDS_CALIBRATED:
        members = firms["in_frame"] & (bands == band)
        if not members.any():
            continue
        propensity = min(1.0, float(hmrc[band]) / float(members.sum()))
        # The double's liability is firm_id thousand pounds per firm.
        expected = float(firms.loc[members, "firm_id"].sum()) * propensity
        assert diagnostics.loc[
            f"hmrc_vat_liability/{band}", "initial_estimate"
        ] == pytest.approx(expected, rel=1e-5)
    registered = firms["vat_registered"]
    np.testing.assert_allclose(
        firms.loc[registered, "vat_liability_k"],
        firms.loc[registered, "firm_id"].astype(float),
    )


def test_firm_rules_take_precedence_over_legacy_evaluator() -> None:
    rules = _ShuffledFirmIdRules()
    config = UKFirmGenerationConfig(
        firm_rules=rules,
        vat_rule_evaluator=_TestVATRuleEvaluator(),
    )
    assert resolve_uk_firm_rules(config) is rules

    legacy = resolve_uk_firm_rules(_firm_config())
    assert isinstance(legacy, LegacyVATEvaluatorRules)
    assert isinstance(legacy.evaluator, _TestVATRuleEvaluator)

    with pytest.raises(ValueError, match="requires a VAT rule evaluator"):
        resolve_uk_firm_rules(UKFirmGenerationConfig())


def test_generation_config_sampling_defaults_and_fast_preset() -> None:
    config = UKFirmGenerationConfig()
    assert config.sample_tail_fraction == 1.0
    assert config.sample_window_fraction == 1.0
    assert config.sample_window_lo_k == 15.0
    assert config.sample_window_hi_k == 150.0
    assert config.sample_cell_floor == 40
    assert config.firm_rules is None
    assert not config.thinned

    fast = UKFirmGenerationConfig.fast(data_vintage="2024-25", seed=9)
    assert fast.sample_window_fraction == FAST_SAMPLE_WINDOW_FRACTION == 0.30
    assert fast.sample_tail_fraction == FAST_SAMPLE_TAIL_FRACTION == 0.05
    assert fast.data_vintage == "2024-25"
    assert fast.vat_threshold == 90.0
    assert fast.seed == 9
    assert fast.thinned

    with pytest.raises(ValueError, match="sample_tail_fraction"):
        UKFirmGenerationConfig(sample_tail_fraction=0.0)
    with pytest.raises(ValueError, match="sample_window_fraction"):
        UKFirmGenerationConfig(sample_window_fraction=1.5)
    with pytest.raises(ValueError, match="sample_window_lo_k"):
        UKFirmGenerationConfig(sample_window_lo_k=200.0)
    with pytest.raises(ValueError, match="sample_cell_floor"):
        UKFirmGenerationConfig(sample_cell_floor=-1)


def test_hmrc_band_columns_are_the_firm_universe_band_names() -> None:
    assert HMRC_BAND_COLUMNS is HMRC_BAND_NAMES


def test_every_generator_vintage_has_a_rule_period() -> None:
    assert set(UK_FIRM_RULE_PERIOD_BY_VINTAGE) == set(firm_generation.VINTAGES)


def test_with_vintage_resolves_the_vintage_threshold() -> None:
    default = UKFirmGenerationConfig(seed=7, n_iterations=5)
    assert default.vat_threshold == 85.0

    # The gotcha with_vintage exists for: replace() keeps the old threshold.
    replaced = dataclasses.replace(default, data_vintage="2024-25")
    assert replaced.vat_threshold == 85.0

    switched = default.with_vintage("2024-25")
    assert switched.data_vintage == "2024-25"
    assert switched.vat_threshold == 90.0
    assert switched.seed == 7
    assert switched.n_iterations == 5
    assert switched.with_vintage("2023-24").vat_threshold == 85.0
    # An explicit counterfactual threshold is still accepted.
    assert UKFirmGenerationConfig(vat_threshold=100.0).vat_threshold == 100.0

    with pytest.raises(ValueError, match="Unknown UK firm vintage"):
        default.with_vintage("1999-00")


def test_generated_firms_convert_to_a_policyengine_uk_firm_table() -> None:
    result = generate_uk_firm_population(
        _source_data(),
        _firm_config(data_vintage="2024-25", n_iterations=5, seed=3),
    )
    firms = result.firms

    table = to_policyengine_uk_firm_table(
        firms, threshold_k=float(result.config.vat_threshold)
    )

    validate_uk_firm_table(table)
    assert len(table) == len(firms)
    assert "UNKNOWN" not in set(table["firm_sic_section"])
    by_id = firms.set_index("firm_id").loc[table["firm_id"]]
    np.testing.assert_array_equal(table["firm_weight"], by_id["firm_weight"])
    np.testing.assert_array_equal(table["base_weight"], by_id["base_weight"])
    np.testing.assert_array_equal(
        table["source_firm_id"], by_id["source_firm_id"].to_numpy(np.int64)
    )
    assert set(table["firm_id"]) == set(firms["firm_id"])


def test_thinned_generation_carries_base_weights_into_the_solve() -> None:
    data = _source_data(scale=60)
    config = UKFirmGenerationConfig.fast(
        vat_rule_evaluator=_TestVATRuleEvaluator(),
        data_vintage="2024-25",
        sample_cell_floor=2,
        n_iterations=3,
        seed=13,
    )

    result = generate_uk_firm_population(data, config)
    firms = result.firms
    frame = firms[firms["in_frame"]]
    drawn_rows = 7 * 60

    assert len(frame) < drawn_rows
    assert (frame["base_weight"] >= 1.0).all()
    assert (frame["base_weight"] > 1.0).any()
    # Base weights reproduce the full draw: the stratum base-weighted totals
    # add up to every drawn frame row.
    assert float(frame["base_weight"].sum()) == pytest.approx(drawn_rows)
    assert (firms.loc[~firms["in_frame"], "base_weight"] == 1.0).all()
    np.testing.assert_array_equal(
        result.calibration.initial_weights,
        firms["base_weight"].to_numpy(),
    )
    # ``initial_weights`` is echoed back by ``solve_firm_weights``; prove the
    # solver itself started from the base weights. Its own first-epoch loss is
    # the loss at ``matrix @ base_weight`` ...
    assert result.calibration.loss_trajectory[0] == pytest.approx(
        result.calibration.initial_loss, rel=1e-5
    )
    # ... and a three-epoch solve stays at the base weights (up to 20 here),
    # far from the all-ones start an unthinned solve uses.
    assert firms["base_weight"].max() > 2.0
    np.testing.assert_allclose(
        firms["firm_weight"].to_numpy(),
        firms["base_weight"].to_numpy(),
        rtol=1e-3,
    )
    diagnostics = result.target_diagnostics.set_index("target_name")
    assert diagnostics.loc[
        "ons_firm_population/total", "initial_estimate"
    ] == pytest.approx(drawn_rows)

    again = generate_uk_firm_population(data, config).firms
    pd.testing.assert_frame_equal(firms, again)


def test_generate_base_firms_draws_inside_half_open_source_bands() -> None:
    torch.manual_seed(0)
    ons_turnover = pd.DataFrame(
        [
            {"SIC Code": 45, "Description": "Retail", "50-99": 500, "5000+": 500},
        ]
    )

    sic, turnover = generate_base_firms(ons_turnover)

    values = turnover.numpy()
    assert sic.tolist() == [45] * 1_000
    assert ((values[:500] >= 50.0) & (values[:500] < 100.0)).all()
    assert ((values[500:] >= 5_000.0) & (values[500:] < 50_000.0)).all()
    # The open top band is log-uniform: its median sits near sqrt(5m x 50m).
    assert np.median(values[500:]) == pytest.approx(15_811.0, rel=0.15)


def test_generate_input_values_uses_recentred_input_ratio() -> None:
    torch.manual_seed(0)
    n_firms = 20_000
    turnover = torch.full((n_firms,), 100.0)
    sic = torch.full((n_firms,), 45, dtype=torch.int64)  # no sector shift

    ratios = (generate_input_values(turnover, sic) / turnover).numpy()

    assert ratios.min() >= INPUT_RATIO_LOWER - 1e-6
    assert ratios.max() <= INPUT_RATIO_UPPER + 1e-6
    # 0.2 + Beta(4, 2) * 0.6 has mean 0.6; value added is always positive.
    assert ratios.mean() == pytest.approx(0.6, abs=0.02)
    zero = generate_input_values(torch.tensor([0.0]), torch.tensor([50]))
    assert zero.tolist() == [0.0]


def test_assign_employment_is_sector_conditional() -> None:
    ons_employment = pd.DataFrame(
        [
            {"SIC Code": 1, "Description": "Small", "0-4": 5, "250+": 0},
            {"SIC Code": 2, "Description": "Large", "0-4": 0, "250+": 5},
            {"SIC Code": "", "Description": "Total", "0-4": 5, "250+": 5},
        ]
    )
    torch.manual_seed(0)
    sic = torch.tensor([1] * 50 + [2] * 50 + [3] * 200, dtype=torch.int64)

    employment = assign_employment(sic, ons_employment).numpy()

    assert employment.shape == (300,)
    assert ((employment[:50] >= 1) & (employment[:50] <= 4)).all()
    assert (employment[50:100] >= 250).all()
    # A sector without an ONS row draws from the national shares.
    unknown = employment[100:]
    assert (unknown <= 4).any() and (unknown >= 250).any()
    assert assign_employment(sic[:0], ons_employment).shape == (0,)


def test_large_employment_slices_hit_the_mean_inside_the_band() -> None:
    shape = large_employment_pareto_shape()
    # DBT's 250-499 / 500-999 / 1000+ shares are 0.51 / 0.26 / 0.23.
    tail = 1.0 - (250.0 / LARGE_EMPLOYMENT_CAP) ** shape
    above_1000 = ((250.0 / 1_000.0) ** shape - (1.0 - tail)) / tail
    assert 1.0 < shape < 1.3
    assert above_1000 == pytest.approx(0.234, abs=0.04)
    for masses in (np.ones(1), np.ones(2_000), np.linspace(0.5, 50.0, 997)):
        means = large_employment_slice_means(masses)
        assert np.all(np.diff(means) > 0)
        assert means.min() >= 250.0
        assert means.max() <= LARGE_EMPLOYMENT_CAP
        weighted = float((masses * means).sum() / masses.sum())
        assert weighted == pytest.approx(LARGE_EMPLOYMENT_MEAN, rel=1e-9)
    # A zero-mass slice takes its quantile rather than dividing by zero.
    assert np.isfinite(large_employment_slice_means(np.array([1.0, 0.0, 1.0]))).all()
    with pytest.raises(ValueError, match="non-negative"):
        large_employment_slice_means(np.array([-1.0, 1.0]))


def test_assign_employment_weights_the_large_band_mean() -> None:
    ons_employment = pd.DataFrame(
        [{"SIC Code": 2, "Description": "Large", "0-4": 0, "250+": 5}]
    )
    torch.manual_seed(0)
    sic = torch.full((400,), 2, dtype=torch.int64)
    weights = np.linspace(1.0, 40.0, 400)

    employment = assign_employment(
        sic,
        ons_employment,
        weights=weights,
        large_employment_mean=LARGE_EMPLOYMENT_MEAN,
    ).numpy()
    torch.manual_seed(0)
    legacy = assign_employment(sic, ons_employment, weights=weights).numpy()

    assert employment.min() >= 250 and employment.max() > 2_000
    weighted = float((weights * employment).sum() / weights.sum())
    assert weighted == pytest.approx(LARGE_EMPLOYMENT_MEAN, rel=1e-3)
    # The default keeps the paper's draw, clamped to [250, 2000], and both
    # rank the rows by the same lognormal draw.
    assert legacy.min() >= 250 and legacy.max() <= 2_000
    inside = (legacy > 250) & (legacy < 2_000)
    order = np.argsort(legacy[inside], kind="stable")
    ranked = employment[inside][order]
    distinct = np.diff(legacy[inside][order]) > 0
    assert np.all(np.diff(ranked)[distinct] >= 0)
    with pytest.raises(ValueError, match="large_employment_mean"):
        UKFirmGenerationConfig(large_employment_mean=100.0)


def test_employment_band_index_array_matches_band_names() -> None:
    employment = np.arange(1, 3_001)
    indices = employment_band_index_array(employment)
    names = [employment_band_name(value) for value in employment]
    order = ["0-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+"]
    assert [order[index] for index in indices] == names


class _TestVATRuleEvaluator:
    def net_liability_k(
        self,
        *,
        turnover_k: np.ndarray,
        input_cost_k: np.ndarray,
        vat_registered: np.ndarray,
        data_vintage: str,
    ) -> np.ndarray:
        del data_vintage
        return np.where(vat_registered, 0.2 * (turnover_k - input_cost_k), 0.0)


class _ShuffledFirmIdRules:
    """Rules double: liability is firm_id thousand pounds, returned reversed."""

    def __init__(self) -> None:
        self.periods: list[int] = []

    def evaluate(self, firms: pd.DataFrame, *, period: int) -> UKFirmRuleOutputs:
        self.periods.append(period)
        firm_id = firms["firm_id"].to_numpy(np.int64)[::-1]
        return UKFirmRuleOutputs(
            firm_id=firm_id,
            vat_liability_if_registered=firm_id.astype(np.float64) * 1000.0,
            provenance={"rules": "test-double"},
        )


def _firm_config(**overrides) -> UKFirmGenerationConfig:
    return UKFirmGenerationConfig(
        vat_rule_evaluator=_TestVATRuleEvaluator(),
        **overrides,
    )


def _source_data(scale: int = 1):
    frames = _source_frames()
    if scale != 1:
        for key in ("ons_turnover", "ons_employment"):
            frame = frames[key]
            numeric = [
                column
                for column in frame.columns
                if column not in {"SIC Code", "Description"}
            ]
            frames[key] = frame.assign(**{c: frame[c] * scale for c in numeric})
    return uk_firm_source_data_from_frames(**frames)


def _ledger_target_profile_mapping(**overrides: str) -> dict[str, object]:
    record_sets = {
        "ons_turnover": DEFAULT_UK_FIRM_TARGET_PROFILE.ons_turnover_record_set,
        "ons_employment": DEFAULT_UK_FIRM_TARGET_PROFILE.ons_employment_record_set,
        "hmrc_population_band": (
            DEFAULT_UK_FIRM_TARGET_PROFILE.hmrc_population_band_record_set
        ),
        "hmrc_liability_band": (
            DEFAULT_UK_FIRM_TARGET_PROFILE.hmrc_liability_band_record_set
        ),
        "ons_sic_turnover": (
            DEFAULT_UK_FIRM_TARGET_PROFILE.ons_sic_turnover_record_set
        ),
        "ons_sic_employment": (
            DEFAULT_UK_FIRM_TARGET_PROFILE.ons_sic_employment_record_set
        ),
        "hmrc_population_sic": (
            DEFAULT_UK_FIRM_TARGET_PROFILE.hmrc_population_sic_record_set
        ),
        "hmrc_liability_sic": DEFAULT_UK_FIRM_TARGET_PROFILE.hmrc_liability_sic_record_set,
        **overrides,
    }
    return {
        "targets": [
            {
                "target_id": target_id,
                "ledger_selector": {"record_set_id": record_sets[logical_key]},
            }
            for logical_key, target_id in UK_FIRM_TARGET_IDS.items()
        ]
    }


def _replace_fact_record_set(
    facts: list[dict[str, object]],
    *,
    old_record_set_id: str,
    new_record_set_id: str,
) -> list[dict[str, object]]:
    updated_facts: list[dict[str, object]] = []
    for fact in facts:
        updated = {**fact, "layout": dict(fact["layout"])}
        if updated["layout"]["record_set_id"] == old_record_set_id:
            updated["layout"]["record_set_id"] = new_record_set_id
        updated_facts.append(updated)
    return updated_facts


def _ledger_facts() -> list[dict[str, object]]:
    facts: list[dict[str, object]] = []
    for sic, values in {
        "00011": {
            "0_49k": 2,
            "50_99k": 1,
            "100_249k": 1,
            "250_499k": 0,
            "500_999k": 0,
            "1000_4999k": 0,
            "5000k_plus": 0,
        },
        "00012": {
            "0_49k": 1,
            "50_99k": 0,
            "100_249k": 1,
            "250_499k": 0,
            "500_999k": 1,
            "1000_4999k": 0,
            "5000k_plus": 0,
        },
    }.items():
        for band, value in values.items():
            facts.append(
                _ledger_fact(
                    "ons.uk_business.cy2025.enterprise_count.by_sic_turnover_band",
                    f"sic_{sic}__{band}",
                    "enterprise_count",
                    value,
                    period_type="calendar_year",
                    period=2025,
                    source_name="ons",
                    unit="count",
                    groupby_dimension="uk.firm.sic_code",
                    dimensions={
                        "uk.firm.sic_code": sic,
                        "uk.firm.turnover_band": band,
                    },
                )
            )
    for sic, values in {
        "00011": {
            "0_4": 2,
            "5_9": 1,
            "10_19": 1,
            "20_49": 0,
            "50_99": 0,
            "100_249": 0,
            "250_plus": 0,
        },
        "00012": {
            "0_4": 1,
            "5_9": 1,
            "10_19": 0,
            "20_49": 1,
            "50_99": 0,
            "100_249": 0,
            "250_plus": 0,
        },
    }.items():
        for band, value in values.items():
            facts.append(
                _ledger_fact(
                    "ons.uk_business.cy2025.enterprise_count.by_sic_employment_band",
                    f"sic_{sic}__{band}",
                    "enterprise_count",
                    value,
                    period_type="calendar_year",
                    period=2025,
                    source_name="ons",
                    unit="count",
                    groupby_dimension="uk.firm.sic_code",
                    dimensions={
                        "uk.firm.sic_code": sic,
                        "uk.firm.employment_band": band,
                    },
                )
            )
    for band, value in {
        "0_49k": 2,
        "50_99k": 1,
        "100_249k": 2,
        "250_499k": 0,
        "500_999k": 1,
        "1000_4999k": 0,
        "5000k_plus": 1,
    }.items():
        facts.append(
            _ledger_fact(
                "ons.uk_business.cy2025.enterprise_count.by_turnover_band",
                band,
                "enterprise_count",
                value,
                period_type="calendar_year",
                period=2025,
                source_name="ons",
                unit="count",
                groupby_dimension="uk.firm.annual_turnover",
            )
        )
    for band, value in {
        "0_4": 3,
        "5_9": 2,
        "10_19": 1,
        "20_49": 1,
        "50_99": 0,
        "100_249": 0,
        "250_plus": 0,
    }.items():
        facts.append(
            _ledger_fact(
                "ons.uk_business.cy2025.enterprise_count.by_employment_band",
                band,
                "enterprise_count",
                value,
                period_type="calendar_year",
                period=2025,
                source_name="ons",
                unit="count",
                groupby_dimension="uk.firm.employees",
            )
        )
    for band, value in {
        "negative_or_zero": 3,
        "1_to_threshold": 3,
        "threshold_to_150k": 2,
        "150k_to_300k": 1,
        "300k_to_500k": 0,
        "500k_to_1m": 1,
        "1m_to_10m": 0,
        "greater_than_10m": 0,
    }.items():
        facts.append(
            _ledger_fact(
                "hmrc.vat.fy2024_25.registered_trader_count.by_turnover_band",
                band,
                "vat_registered_trader_count",
                value,
                period_type="fiscal_year",
                period=2024,
                source_name="hmrc",
                unit="count",
                groupby_dimension="uk.firm.annual_turnover",
            )
        )
    for band, value in {
        "negative_or_zero": 0.0,
        "1_to_threshold": 400_000.0,
        "threshold_to_150k": 1_200_000.0,
        "150k_to_300k": 1_500_000.0,
        "300k_to_500k": 0.0,
        "500k_to_1m": 2_000_000.0,
        "1m_to_10m": 0.0,
        "greater_than_10m": 0.0,
    }.items():
        facts.append(
            _ledger_fact(
                "hmrc.vat.fy2024_25.net_liability.by_turnover_band",
                band,
                "net_vat_liability",
                value,
                period_type="fiscal_year",
                period=2024,
                source_name="hmrc",
                unit="gbp",
                groupby_dimension="uk.firm.annual_turnover",
            )
        )
    for sic, value in {"00011": 3, "00012": 2}.items():
        facts.append(
            _ledger_fact(
                "hmrc.vat.fy2024_25.registered_trader_count.by_sic",
                f"sic_{sic}",
                "vat_registered_trader_count",
                value,
                period_type="fiscal_year",
                period=2024,
                source_name="hmrc",
                unit="count",
                groupby_dimension="uk.firm.sic_code",
                dimensions={"uk.firm.sic_code": sic},
            )
        )
    for sic, value in {"00011": 2_000_000.0, "00012": 1_500_000.0}.items():
        facts.append(
            _ledger_fact(
                "hmrc.vat.fy2024_25.net_liability.by_sic",
                f"sic_{sic}",
                "net_vat_liability",
                value,
                period_type="fiscal_year",
                period=2024,
                source_name="hmrc",
                unit="gbp",
                groupby_dimension="uk.firm.sic_code",
                dimensions={"uk.firm.sic_code": sic},
            )
        )
    return facts


def _ledger_fact(
    record_set_id: str,
    band: str,
    measure_id: str,
    value: float,
    *,
    period_type: str,
    period: int,
    source_name: str,
    unit: str,
    groupby_dimension: str,
    dimensions: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "arch.consumer_fact.v1",
        "aggregate_fact_key": f"arch.aggregate_fact.v2:{record_set_id}.{band}",
        "value": value,
        "period": {"type": period_type, "value": period},
        "geography": {"level": "country", "id": "K02000001"},
        "entity": {"name": "firm"},
        "observed_measure": {
            "source_name": source_name,
            "source_measure_id": measure_id,
            "unit": unit,
        },
        "dimensions": dimensions or {},
        "aggregation": {"method": "sum"},
        "layout": {
            "record_set_id": record_set_id,
            "groupby_dimension": groupby_dimension,
            "groupby_value_id": band,
            "measure_id": measure_id,
        },
    }


def _source_frames() -> dict[str, pd.DataFrame]:
    ons_turnover = pd.DataFrame(
        [
            {
                "SIC Code": 11,
                "Description": "Manufacturing",
                "0-49": 2,
                "50-99": 1,
                "100-249": 1,
                "250-499": 0,
                "500-999": 0,
                "1000-4999": 0,
                "5000+": 0,
                "Total": 4,
            },
            {
                "SIC Code": 12,
                "Description": "Wholesale",
                "0-49": 1,
                "50-99": 0,
                "100-249": 1,
                "250-499": 0,
                "500-999": 1,
                "1000-4999": 0,
                "5000+": 0,
                "Total": 3,
            },
            {
                "SIC Code": "",
                "Description": "Total",
                "0-49": 3,
                "50-99": 1,
                "100-249": 2,
                "250-499": 0,
                "500-999": 1,
                "1000-4999": 0,
                "5000+": 0,
                "Total": 7,
            },
        ]
    )
    ons_employment = pd.DataFrame(
        [
            {
                "SIC Code": 11,
                "Description": "Manufacturing",
                "0-4": 2,
                "5-9": 1,
                "10-19": 1,
                "20-49": 0,
                "50-99": 0,
                "100-249": 0,
                "250+": 0,
            },
            {
                "SIC Code": 12,
                "Description": "Wholesale",
                "0-4": 1,
                "5-9": 1,
                "10-19": 0,
                "20-49": 1,
                "50-99": 0,
                "100-249": 0,
                "250+": 0,
            },
            {
                "SIC Code": "",
                "Description": "Total",
                "0-4": 3,
                "5-9": 2,
                "10-19": 1,
                "20-49": 1,
                "50-99": 0,
                "100-249": 0,
                "250+": 0,
            },
        ]
    )
    hmrc_population_band = pd.DataFrame(
        [
            {
                "Financial_Year": "2023-24",
                "Negative_or_Zero": 2,
                "£1_to_Threshold": 2,
                "£Threshold_to_£150k": 4,
                "£150k_to_£300k": 2,
                "£300k_to_£500k": 0,
                "£500k_to_£1m": 1,
                "£1m_to_£10m": 0,
                "Greater_than_£10m": 0,
            },
            {
                "Financial_Year": "2024-25",
                "Negative_or_Zero": 3,
                "£1_to_Threshold": 3,
                "£Threshold_to_£150k": 2,
                "£150k_to_£300k": 1,
                "£300k_to_£500k": 0,
                "£500k_to_£1m": 1,
                "£1m_to_£10m": 0,
                "Greater_than_£10m": 0,
            },
        ]
    )
    hmrc_population_sector = pd.DataFrame(
        [
            {"Trade_Sector": 11, "2023-24": 2, "2024-25": 3},
            {"Trade_Sector": 12, "2023-24": 4, "2024-25": 2},
            {"Trade_Sector": "Total", "2023-24": 6, "2024-25": 5},
        ]
    )
    hmrc_liability_band = pd.DataFrame(
        [
            {
                "Financial_Year": "2023-24",
                "Negative_or_Zero": 0.0,
                "£1_to_Threshold": 0.3,
                "£Threshold_to_£150k": 0.9,
                "£150k_to_£300k": 1.3,
                "£300k_to_£500k": 0.0,
                "£500k_to_£1m": 1.8,
                "£1m_to_£10m": 0.0,
                "Greater_than_£10m": 0.0,
            },
            {
                "Financial_Year": "2024-25",
                "Negative_or_Zero": 0.0,
                "£1_to_Threshold": 0.4,
                "£Threshold_to_£150k": 1.2,
                "£150k_to_£300k": 1.5,
                "£300k_to_£500k": 0.0,
                "£500k_to_£1m": 2.0,
                "£1m_to_£10m": 0.0,
                "Greater_than_£10m": 0.0,
            },
        ]
    )
    hmrc_liability_sector = pd.DataFrame(
        [
            {"Trade_Sector": 11, "2023-24": 1.8, "2024-25": 2.0},
            {"Trade_Sector": 12, "2023-24": 1.4, "2024-25": 1.5},
            {"Trade_Sector": "Total", "2023-24": 3.2, "2024-25": 3.5},
        ]
    )
    assert set(HMRC_BAND_COLUMNS).issubset(hmrc_population_band.columns)
    assert set(HMRC_BAND_COLUMNS).issubset(hmrc_liability_band.columns)
    return {
        "ons_turnover": ons_turnover,
        "ons_employment": ons_employment,
        "hmrc_population_band": hmrc_population_band,
        "hmrc_population_sector": hmrc_population_sector,
        "hmrc_liability_band": hmrc_liability_band,
        "hmrc_liability_sector": hmrc_liability_sector,
    }
