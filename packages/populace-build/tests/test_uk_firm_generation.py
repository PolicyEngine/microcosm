from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from populace.build.uk_runtime import (
    AxiomVATRuleEvaluator as PublicAxiomVATRuleEvaluator,
)
from populace.build.uk_runtime.firm_generation import (
    DEFAULT_UK_FIRM_TARGET_PROFILE,
    HMRC_BAND_COLUMNS,
    INPUT_FILES,
    UK_FIRM_TARGET_IDS,
    AxiomVATRuleEvaluator,
    UKFirmGenerationConfig,
    calculate_vat_liability_values,
    employment_band_name,
    generate_uk_firm_population,
    hmrc_band_name,
    read_uk_firm_source_data,
    uk_firm_source_data_from_frames,
    uk_firm_source_data_from_ledger_facts,
    uk_firm_target_profile_from_mapping,
    write_uk_firm_population,
)


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
    assert diagnostics.loc["hmrc_vat_population_sector/11", "target"] == 3.0
    assert diagnostics.loc["hmrc_vat_population_sector/12", "target"] == 2.0
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
        "vat_registered",
        "generation_status",
    ]
    assert firms["firm_id"].tolist() == sorted(firms["firm_id"].tolist())
    assert firms["firm_id"].is_unique
    assert firms["source_firm_key"].is_unique
    assert set(firms["generation_status"]) == {"experimental"}
    assert (firms["firm_weight"] > 0).all()
    assert firms.loc[
        firms["annual_turnover_k"] > config.vat_threshold,
        "vat_registered",
    ].all()
    zero_turnover = firms[firms["annual_turnover_k"] == 0]
    assert len(zero_turnover) == 3
    assert zero_turnover["vat_registered"].all()
    assert 0.0 <= result.validation.overall <= 1.0
    assert result.calibration.final_loss <= result.calibration.initial_loss
    assert len(result.calibration.loss_trajectory) == config.n_iterations
    assert result.target_diagnostics["target_name"].str.startswith(
        ("hmrc_", "ons_")
    ).all()
    assert "initial_estimate" in result.target_diagnostics.columns

    base_firms = firms.iloc[: len(result.calibration.weights)].copy()
    base_firms["employment_band"] = base_firms["employment"].apply(
        employment_band_name
    )
    weighted_employment = base_firms.groupby("employment_band")["firm_weight"].sum()
    diagnostics = result.target_diagnostics.set_index("target_name")
    for band in ("0-4", "5-9", "10-19", "20-49"):
        assert diagnostics.loc[
            f"ons_firm_employment/{band}",
            "estimate",
        ] == pytest.approx(float(weighted_employment.get(band, 0.0)))


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
    assert targets_2023["hmrc_vat_population_sector/11"] == 2.0
    assert targets_2024["hmrc_vat_population_sector/11"] == 3.0


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
    result = calculate_vat_liability_values(
        _firm_config(),
        torch.tensor([100.0, 100.0], dtype=torch.float32),
        torch.tensor([40.0, 120.0], dtype=torch.float32),
        torch.tensor([True, True], dtype=torch.bool),
    )

    assert result.tolist() == pytest.approx([12.0, -4.0])


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


def _firm_config(**overrides) -> UKFirmGenerationConfig:
    return UKFirmGenerationConfig(
        vat_rule_evaluator=_TestVATRuleEvaluator(),
        **overrides,
    )


def _source_data():
    return uk_firm_source_data_from_frames(**_source_frames())


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
            }
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
            }
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
