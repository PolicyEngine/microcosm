"""Versioned Populace Section 199A simulation tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from populace.build.us_runtime import qbi_inputs as qbi_inputs_module
from populace.build.us_runtime.qbi_inputs import US_QBI_OUTPUT_COLUMNS
from populace.build.us_runtime.qbi_simulation import (
    QBI_SIMULATION_SOURCE_NAMES,
    QBI_SIMULATION_V2,
    QBI_SIMULATION_V3,
    QBI_SIMULATION_VERSION,
    QbiSimulationInputs,
    load_qbi_simulation_assumptions,
    load_sstb_crosswalk,
    parse_qbi_simulation_assumptions,
    parse_sstb_crosswalk,
    qbi_simulation_summary,
    simulate_qbi_inputs,
    simulate_qbi_v3_wage_capital,
    us_qbi_simulation_stage_spec,
    with_qbi_simulation_from_puf_arrays,
)

_PUF_2024_PATH = os.environ.get("POPULACE_PUF_2024_H5")
requires_puf_2024 = pytest.mark.skipif(
    not _PUF_2024_PATH or not Path(_PUF_2024_PATH).is_file(),
    reason="set POPULACE_PUF_2024_H5 to the restricted pinned artifact",
)


def _sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _synthetic_sources() -> dict[str, np.ndarray]:
    index = np.arange(64, dtype=np.float64)
    return {
        "self_employment_income": np.where(
            index % 3 == 0,
            (index + 1) * 1_000,
            np.where(index % 7 == 0, -(index + 1) * 400, 0),
        ),
        "farm_operations_income": np.where(
            index % 5 == 0,
            (index + 2) * 700,
            np.where(index % 11 == 0, -(index + 2) * 250, 0),
        ),
        "farm_rent_income": np.where(
            index % 8 == 0,
            (index + 3) * 350,
            0,
        ),
        "rental_income": np.where(
            index % 4 == 0,
            (index + 4) * 900,
            np.where(index % 13 == 0, -(index + 4) * 300, 0),
        ),
        "estate_income": np.where(
            index % 9 == 0,
            (index + 5) * 1_100,
            0,
        ),
        "partnership_s_corp_income": np.where(
            index % 6 == 0,
            (index + 6) * 1_500,
            np.where(index % 10 == 0, -(index + 6) * 500, 0),
        ),
        "non_qualified_dividend_income": np.where(
            index % 2 == 0,
            (index + 1) * 120,
            0,
        ),
    }


def _v2_payload() -> dict[str, object]:
    resource = files("populace.build.us").joinpath("qbi_assumptions_v2.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _synthetic_sstb_crosswalk_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "crosswalk_version": "synthetic-test-v1",
        "status": "live",
        "meta": {
            "industry_vintage": "synthetic Census industry",
            "occupation_vintage": "synthetic Census occupation",
            "legal_basis": "synthetic Section 199A test basis",
            "wiring_notes": ["synthetic test wiring"],
            "sstb_category_values": ["law"],
        },
        "industry_2017": [
            {
                "census_code": "4040",
                "census_title": "Synthetic non-SSTB industry",
                "naics": "00",
                "sstb_category": ["law"],
                "classification": "non_sstb",
                "probability": 0.0,
                "rationale": "Synthetic deterministic non-SSTB entry",
            },
        ],
        "industry_explicit_nonsstb_neighbors": [
            {
                "census_code": "0000",
                "census_title": "Synthetic documented industry",
                "why": "Synthetic documentation row",
                "probability": 0.0,
            }
        ],
        "occupation_2018": [
            {
                "census_code": "1010",
                "census_title": "Synthetic clear occupation",
                "soc": "00-0001",
                "sstb_category": ["law"],
                "classification": "clear_sstb",
                "probability": 1.0,
                "rationale": "Synthetic deterministic SSTB entry",
            },
            {
                "census_code": "2020",
                "census_title": "Synthetic non-SSTB occupation",
                "soc": "00-0002",
                "sstb_category": ["law"],
                "classification": "non_sstb",
                "probability": 0.0,
                "rationale": "Synthetic deterministic non-SSTB entry",
            },
            {
                "census_code": "3030",
                "census_title": "Synthetic ambiguous occupation",
                "soc": "00-0003",
                "sstb_category": ["law"],
                "classification": "ambiguous",
                "probability": 0.3,
                "rationale": "Synthetic ambiguous entry",
                "provisional": True,
                "basis": "Synthetic provisional basis",
            },
        ],
        "occupation_explicit_nonsstb_notes": [
            {
                "census_code": "0000",
                "census_title": "Synthetic documented occupation",
                "why": "Synthetic documentation row",
                "probability": 0.0,
            }
        ],
    }


def test_v1_assumptions_pin_stream_order_models_and_manifest_marker() -> None:
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_VERSION)

    assert assumptions.qbi_simulation_version == 1
    assert assumptions.engine == "archived_source_weighted_v1"
    assert assumptions.bit_generator == "PCG64"
    assert (
        assumptions.qualification_seed,
        assumptions.w2_ubia_seed,
        assumptions.investment_seed,
        assumptions.sstb_seed,
    ) == (41, 42, 43, 64)
    assert assumptions.source_order == QBI_SIMULATION_SOURCE_NAMES
    assert assumptions.sstb_source_order == (
        "self_employment_income",
        "partnership_s_corp_income",
        "estate_income",
    )
    assert tuple(exposure.source for exposure in assumptions.reit_ptp_exposures) == (
        "non_qualified_dividend_income",
        "partnership_s_corp_income",
    )
    assert tuple(exposure.source for exposure in assumptions.bdc_exposures) == (
        "non_qualified_dividend_income",
    )

    stage = us_qbi_simulation_stage_spec()
    assert any(
        artifact.get("populace_qbi_simulation_version") == 1
        for artifact in stage.artifacts
    )
    assert set(US_QBI_OUTPUT_COLUMNS) <= set(stage.outputs)


def test_v2_assumptions_pin_derivations_host_columns_and_family_seeds() -> None:
    v1 = load_qbi_simulation_assumptions(QBI_SIMULATION_VERSION)
    v2 = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)

    assert v2.schema_version == 2
    assert v2.qbi_simulation_version == 2
    assert (
        v2.qualification_seed,
        v2.sstb_seed,
        v2.w2_seed,
        v2.ubia_seed,
        v2.investment_seed,
    ) == (2041, 2064, 2042, 2044, 2043)
    assert {
        source: derivation.mode
        for source, derivation in v2.qualification_by_source.items()
    } == {
        "self_employment_income": "derived",
        "farm_operations_income": "derived",
        "farm_rent_income": "prior",
        "rental_income": "prior",
        "estate_income": "prior",
        "partnership_s_corp_income": "prior",
    }
    assert {
        source: derivation.prior_probability
        for source, derivation in v2.qualification_by_source.items()
        if derivation.mode == "prior"
    } == {
        "farm_rent_income": 0.8,
        "rental_income": 0.7,
        "estate_income": 0.6,
        "partnership_s_corp_income": 0.9,
    }
    assert all(derivation.rationale for derivation in v2.qualification_derivations)
    rationales = {
        source: derivation.rationale
        for source, derivation in v2.qualification_by_source.items()
    }
    assert "positive Schedule C income" in rationales["self_employment_income"]
    assert "active farm trade-or-business" in rationales["farm_operations_income"]
    assert "trade or business" in rationales["farm_rent_income"]
    assert "safe-harbor" in rationales["rental_income"]
    assert "weakest residual prior" in rationales["estate_income"]
    assert "v3" in rationales["estate_income"]
    assert "guaranteed-payment" in rationales["partnership_s_corp_income"]
    assert "reasonable-compensation" in rationales["partnership_s_corp_income"]
    assert "53%" in rationales["partnership_s_corp_income"]
    assert "17%" in rationales["partnership_s_corp_income"]
    assert v2.sstb_classification.mode == "crosswalk"
    assert v2.sstb_classification.occupation_column == "PEIOOCC"
    assert v2.sstb_classification.industry_column is None
    assert v2.sstb_classification.agi_column == "AGI"
    assert {
        band.label: band.probability
        for band in v2.sstb_classification.passive_passthrough_sstb_prior_by_agi
    } == {
        "-inf:200000": 0.264,
        "200000:inf": 0.17,
    }
    assert "$58.24B / $221.00B = 0.2635" in v2.sstb_classification.rationale
    assert "$37.54B / $221.00B = 0.1699" in v2.sstb_classification.rationale
    assert v2.reit_ptp_anchor.provisional
    assert v2.reit_ptp_anchor.published_income_dollars == 21_070_000_000.0
    assert v2.reit_ptp_anchor.published_component_dollars == 4_200_000_000.0
    assert v2.reit_ptp_anchor.comparison_component_2022_dollars == 2_900_000_000.0
    assert v2.reit_ptp_anchor.replay_factor_band == (0.3, 3.0)
    assert v2.bdc_anchor.provisional
    assert v2.bdc_anchor.published_income_dollars is None
    assert v2.bdc_anchor.replay_factor_band is None
    assert v2.w2_model == v1.w2_model
    assert v2.profit_margin_parameters == v1.profit_margin_parameters
    assert v2.has_employees_slope_per_dollar == v1.has_employees_slope_per_dollar
    assert v2.has_employees_target_share == v1.has_employees_target_share
    assert v2.intercept_bisection_iterations == v1.intercept_bisection_iterations
    assert v2.labor_ratio_parameters == v1.labor_ratio_parameters
    assert v2.ubia_model == v1.ubia_model
    assert v2.ubia_sigma == v1.ubia_sigma
    assert v2.ubia_multiples == v1.ubia_multiples
    assert v2.capital_intensity_probabilities == v1.capital_intensity_probabilities
    assert {
        exposure.source: exposure.probability_of_receiving
        for exposure in v2.reit_ptp_exposures
    } == {
        "non_qualified_dividend_income": 0.35,
        "partnership_s_corp_income": 0.09,
    }
    assert v2.bdc_exposures == v1.bdc_exposures


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (
            lambda payload: payload.__setitem__("unknown_root_key", True),
            "keys must match",
        ),
        (
            lambda payload: payload["qualification_derivations"][
                "self_employment_income"
            ].__setitem__("mode", "coin_flip"),
            "Unknown QBI v2 qualification mode",
        ),
        (
            lambda payload: payload["sstb_classification"].__setitem__(
                "mode", "flat_bernoulli"
            ),
            "Unknown QBI v2 SSTB classification mode",
        ),
        (
            lambda payload: payload["rng"]["seeds"].__setitem__("extra", 99),
            "keys must match",
        ),
        (
            lambda payload: payload["rng"]["seeds"].__setitem__("qualification", -1),
            "family seeds must be nonnegative",
        ),
        (
            lambda payload: payload["rng"]["seeds"].__setitem__(
                "ubia", payload["rng"]["seeds"]["w2"]
            ),
            "family seeds must be distinct",
        ),
        (
            lambda payload: payload["sstb_classification"][
                "passive_passthrough_sstb_prior_by_agi"
            ].__setitem__("200001:inf", 0.0),
            "contiguous and non-overlapping",
        ),
    ),
)
def test_v2_schema_rejects_unknown_keys_modes_and_invalid_bands(
    mutate,
    match: str,
) -> None:
    payload = copy.deepcopy(_v2_payload())
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        parse_qbi_simulation_assumptions(
            payload,
            qbi_simulation_version=QBI_SIMULATION_V2,
        )


def test_sstb_crosswalk_placeholder_fails_closed_and_live_fixture_loads() -> None:
    with pytest.raises(ValueError, match="status is 'placeholder'"):
        load_sstb_crosswalk("sstb_crosswalk_placeholder.json")

    crosswalk = parse_sstb_crosswalk(_synthetic_sstb_crosswalk_payload())

    assert crosswalk.status == "live"
    assert crosswalk.mapping_for("occupation") == {
        1010: 1.0,
        2020: 0.0,
        3030: 0.3,
    }


def test_caller_constructed_crosswalk_still_fails_closed() -> None:
    crosswalk = parse_sstb_crosswalk(_synthetic_sstb_crosswalk_payload())
    malformed = replace(
        crosswalk,
        occupation_entries=(
            replace(
                crosswalk.occupation_entries[0],
                classification="unknown_classification",
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown classification"):
        simulate_qbi_inputs(
            QbiSimulationInputs.from_puf_arrays(_synthetic_sources()),
            assumptions=load_qbi_simulation_assumptions(QBI_SIMULATION_V2),
            qbi_simulation_version=QBI_SIMULATION_V2,
            sstb_crosswalk=malformed,
        )


def test_v2_simulation_fails_closed_before_using_placeholder_crosswalk() -> None:
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)
    assumptions = replace(
        assumptions,
        sstb_classification=replace(
            assumptions.sstb_classification,
            crosswalk_resource="sstb_crosswalk_placeholder.json",
        ),
    )

    with pytest.raises(ValueError, match="status is 'placeholder'"):
        simulate_qbi_inputs(
            inputs,
            assumptions=assumptions,
            qbi_simulation_version=QBI_SIMULATION_V2,
        )


def test_v2_derives_law_determined_flags_and_draws_only_residual_priors() -> None:
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)
    outputs = simulate_qbi_inputs(
        inputs,
        assumptions=assumptions,
        qbi_simulation_version=QBI_SIMULATION_V2,
        sstb_crosswalk=_synthetic_sstb_crosswalk_payload(),
    )
    qualification_rng = np.random.default_rng(assumptions.qualification_seed)

    for derivation in assumptions.qualification_derivations:
        if derivation.mode == "derived":
            expected = inputs.source(derivation.source) != 0.0
        else:
            assert derivation.prior_probability is not None
            expected = qualification_rng.random(inputs.n) < derivation.prior_probability
        np.testing.assert_array_equal(
            outputs[f"{derivation.source}_would_be_qualified"],
            expected,
        )

    assert not np.any(outputs["business_is_sstb"])
    assert not np.any(outputs["sstb_self_employment_income_before_lsr"])


def test_v2_qualification_mode_change_preserves_other_family_bytes() -> None:
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)
    prior_partnership = replace(
        assumptions,
        qualification_derivations=tuple(
            replace(
                derivation,
                mode="prior",
                prior_probability=1.0,
            )
            if derivation.source == "partnership_s_corp_income"
            else derivation
            for derivation in assumptions.qualification_derivations
        ),
    )
    fixture = _synthetic_sstb_crosswalk_payload()
    baseline = simulate_qbi_inputs(
        inputs,
        assumptions=assumptions,
        qbi_simulation_version=QBI_SIMULATION_V2,
        sstb_crosswalk=fixture,
    )
    changed = simulate_qbi_inputs(
        inputs,
        assumptions=prior_partnership,
        qbi_simulation_version=QBI_SIMULATION_V2,
        sstb_crosswalk=fixture,
    )

    assert not np.array_equal(
        baseline["partnership_s_corp_income_would_be_qualified"],
        changed["partnership_s_corp_income_would_be_qualified"],
    )
    for column in (
        "w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "qualified_reit_and_ptp_income",
        "qualified_bdc_income",
    ):
        assert (
            np.asarray(baseline[column]).tobytes()
            == np.asarray(changed[column]).tobytes()
        )


def test_v2_w2_and_ubia_family_seeds_are_independent() -> None:
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)
    fixture = _synthetic_sstb_crosswalk_payload()

    def run(resolved_assumptions):
        return simulate_qbi_inputs(
            inputs,
            assumptions=resolved_assumptions,
            qbi_simulation_version=QBI_SIMULATION_V2,
            sstb_crosswalk=fixture,
        )

    baseline = run(assumptions)
    changed_w2 = run(replace(assumptions, w2_seed=assumptions.w2_seed + 100))
    changed_ubia = run(replace(assumptions, ubia_seed=assumptions.ubia_seed + 100))

    assert not np.array_equal(
        baseline["w2_wages_from_qualified_business"],
        changed_w2["w2_wages_from_qualified_business"],
    )
    assert np.asarray(baseline["unadjusted_basis_qualified_property"]).tobytes() == (
        np.asarray(changed_w2["unadjusted_basis_qualified_property"]).tobytes()
    )
    assert not np.array_equal(
        baseline["unadjusted_basis_qualified_property"],
        changed_ubia["unadjusted_basis_qualified_property"],
    )
    assert np.asarray(baseline["w2_wages_from_qualified_business"]).tobytes() == (
        np.asarray(changed_ubia["w2_wages_from_qualified_business"]).tobytes()
    )


def test_v3_new_family_streams_are_independent() -> None:
    arrays = {
        name: np.tile(values, 64) for name, values in _synthetic_sources().items()
    }
    inputs = QbiSimulationInputs.from_puf_arrays(arrays)
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V3)

    def run(**seed_change):
        return simulate_qbi_v3_wage_capital(
            inputs,
            assumptions=replace(assumptions, **seed_change),
        )

    baseline = run()
    changed_entity = run(entity_split_seed=assumptions.entity_split_seed + 100)
    changed_industry = run(latent_industry_seed=assumptions.latent_industry_seed + 100)
    changed_employer = run(employer_gate_seed=assumptions.employer_gate_seed + 100)
    changed_margin = run(margin_quantile_seed=assumptions.margin_quantile_seed + 100)
    changed_ubia = run(ubia_dispersion_seed=assumptions.ubia_dispersion_seed + 100)

    np.testing.assert_array_equal(
        baseline.positive_qbi,
        changed_entity.positive_qbi,
    )
    assert not np.array_equal(baseline.legal_form, changed_entity.legal_form)

    for changed in (changed_industry, changed_employer, changed_margin, changed_ubia):
        np.testing.assert_array_equal(baseline.legal_form, changed.legal_form)
        np.testing.assert_array_equal(baseline.positive_qbi, changed.positive_qbi)

    np.testing.assert_array_equal(
        baseline.has_employees,
        changed_industry.has_employees,
    )
    np.testing.assert_array_equal(baseline.receipts, changed_industry.receipts)
    assert not np.array_equal(baseline.w2_wages, changed_industry.w2_wages)
    assert not np.array_equal(baseline.ubia, changed_industry.ubia)

    np.testing.assert_array_equal(baseline.receipts, changed_employer.receipts)
    np.testing.assert_array_equal(baseline.ubia, changed_employer.ubia)
    assert not np.array_equal(
        baseline.has_employees,
        changed_employer.has_employees,
    )
    assert not np.array_equal(baseline.w2_wages, changed_employer.w2_wages)

    np.testing.assert_array_equal(
        baseline.has_employees,
        changed_margin.has_employees,
    )
    assert not np.array_equal(baseline.receipts, changed_margin.receipts)
    assert not np.array_equal(baseline.w2_wages, changed_margin.w2_wages)
    assert not np.array_equal(baseline.ubia, changed_margin.ubia)

    np.testing.assert_array_equal(
        baseline.has_employees,
        changed_ubia.has_employees,
    )
    np.testing.assert_array_equal(baseline.receipts, changed_ubia.receipts)
    np.testing.assert_array_equal(baseline.w2_wages, changed_ubia.w2_wages)
    assert not np.array_equal(baseline.ubia, changed_ubia.ubia)


def test_version_and_random_stream_order_are_strict() -> None:
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_VERSION)
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())

    with pytest.raises(ValueError, match="Unsupported qbi_simulation_version"):
        load_qbi_simulation_assumptions(4)
    with pytest.raises(ValueError, match="does not match assumptions"):
        simulate_qbi_inputs(
            inputs,
            assumptions=assumptions,
            qbi_simulation_version=2,
        )
    with pytest.raises(ValueError, match="random-stream order"):
        replace(
            assumptions,
            source_order=tuple(reversed(assumptions.source_order)),
        ).validate()


def test_raw_puf_e_codes_normalize_to_the_engine_contract() -> None:
    inputs = QbiSimulationInputs.from_puf_arrays(
        {
            "E00900": [10.0, -20.0],
            "E02100": [30.0, -40.0],
            "E27200": [50.0, -60.0],
            "E25850": [80.0, 10.0],
            "E25860": [20.0, 30.0],
            "E26390": [90.0, 10.0],
            "E26400": [40.0, 30.0],
            "E25980": [100.0, 10.0],
            "E25960": [30.0, 20.0],
            "E26190": [50.0, 40.0],
            "E26180": [20.0, 60.0],
            "E00600": [70.0, 90.0],
            "E00650": [20.0, 30.0],
        }
    )

    np.testing.assert_array_equal(inputs.self_employment_income, [10.0, -20.0])
    np.testing.assert_array_equal(inputs.farm_operations_income, [30.0, -40.0])
    np.testing.assert_array_equal(inputs.farm_rent_income, [50.0, -60.0])
    np.testing.assert_array_equal(inputs.rental_income, [60.0, -20.0])
    np.testing.assert_array_equal(inputs.estate_income, [50.0, -20.0])
    np.testing.assert_array_equal(
        inputs.partnership_s_corp_income,
        [100.0, -30.0],
    )
    np.testing.assert_array_equal(
        inputs.non_qualified_dividend_income,
        [50.0, 60.0],
    )


def test_v1_has_golden_seeded_streams_for_all_fifteen_outputs() -> None:
    expected_hashes = {
        "estate_income_would_be_qualified": (
            "7bf4a0ef4e6916f874c23efc606672d61f0f0ccd83058893c6beff1a50a567c1"
        ),
        "farm_operations_income_would_be_qualified": (
            "dc46524410fb5d428d8a512398e4b83bac03c4dbd416c9a7174d01b8519f1c87"
        ),
        "farm_rent_income_would_be_qualified": (
            "8e985d09548b8c18d28ba706a75dd8877b9e210b8b69a95cd13fbd65795f495b"
        ),
        "partnership_s_corp_income_would_be_qualified": (
            "318f8f7a0e55a809b35bb19fa55889b27f0614f1bffdbae2ce213d1ed38daf38"
        ),
        "rental_income_would_be_qualified": (
            "e9d52b86fe42edc3d6935c6a5ea6501cd1fb3ce6bd3d2451671d60131e8cdc66"
        ),
        "self_employment_income_would_be_qualified": (
            "bd503d852bd08747790229cc537ad70b99c9c7fe6757ebc135fc116351db14eb"
        ),
        "sstb_self_employment_income_would_be_qualified": (
            "affa2ccb6317960490062d31722782f839f6a96a9cf2244f80b02761b6b84de3"
        ),
        "business_is_sstb": (
            "aa6b310d2670457604de88303bd67a49a7d0e905ec0784cca74168e78e5159fc"
        ),
        "qualified_bdc_income": (
            "076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560"
        ),
        "sstb_self_employment_income_before_lsr": (
            "ccb21860777bc6982536880d454e1f18bd920d1491c4a4ef8c372b636428ee3e"
        ),
        "sstb_unadjusted_basis_qualified_property": (
            "b656cd1c0b58ea012cbd8a28b830abef3f15a26d42dd0c823e1b595a38413eef"
        ),
        "sstb_w2_wages_from_qualified_business": (
            "076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560"
        ),
        "w2_wages_from_qualified_business": (
            "687685c48717d9f174b885a0d2623dfbb8d99b1a39f456a9a04f7a7913752fdf"
        ),
    }
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())
    outputs = simulate_qbi_inputs(
        inputs,
        assumptions=load_qbi_simulation_assumptions(QBI_SIMULATION_VERSION),
        qbi_simulation_version=QBI_SIMULATION_VERSION,
    )

    # Beta and lognormal draws route through libm, whose last-ulp rounding
    # differs across platforms; these two leaves get value-based goldens with
    # tolerance while every flag/linear leaf stays byte-exact.
    libm_sensitive_expected: dict[str, dict[int, float]] = {
        "qualified_reit_and_ptp_income": {
            2: 37.66626208631762,
            12: 103.79383958799434,
            16: 233.53102781186632,
            26: 28.33837225431461,
            28: 211.32889926679022,
            30: 80.87751309669761,
            32: 111.54473324411704,
            36: 520.9426655795415,
            38: 694.8521547117064,
            48: 899.592629035168,
            50: 236.78819258966882,
            54: 990.4968932479194,
            62: 1361.5004139746575,
        },
        "unadjusted_basis_qualified_property": {
            0: 30518.054316323058,
            4: 86945.18129450442,
            5: 31569.609596577146,
            6: 40689.834303024974,
            8: 14465.183779022262,
            10: 24328.436608781318,
            15: 14165.063201140407,
            16: 248840.89983255556,
            20: 52988.84715631746,
            27: 79734.50293313447,
            32: 168466.50112907932,
            35: 20541.337464914657,
            36: 760501.1618940977,
            40: 384821.58411189757,
            48: 523570.5769025755,
            50: 68223.48718574771,
            52: 832480.8658452359,
            54: 361472.420883897,
            55: 18378.210763760664,
            56: 29930.640951148995,
        },
    }

    assert tuple(outputs) == US_QBI_OUTPUT_COLUMNS
    assert {
        column: _sha256(np.asarray(values))
        for column, values in outputs.items()
        if column in expected_hashes
    } == expected_hashes
    for column, sparse_expected in libm_sensitive_expected.items():
        expected = np.zeros(64, dtype=np.float64)
        for index, value in sparse_expected.items():
            expected[index] = value
        actual = np.asarray(outputs[column], dtype=np.float64)
        assert (actual != 0.0).tolist() == (expected != 0.0).tolist()
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)


def test_stage_wrapper_replaces_stale_leaves_and_preserves_sstb_identities() -> None:
    sources = _synthetic_sources()
    original_self_employment = sources["self_employment_income"].copy()
    sources["w2_wages_from_qualified_business"] = np.full(64, -1.0)
    sources["business_is_sstb"] = np.ones(64, dtype=bool)

    result = with_qbi_simulation_from_puf_arrays(
        sources,
        qbi_simulation_version=QBI_SIMULATION_VERSION,
    )
    is_sstb = np.asarray(result["business_is_sstb"], dtype=bool)

    assert not np.any(
        np.asarray(result["self_employment_income_would_be_qualified"])
        & np.asarray(result["sstb_self_employment_income_would_be_qualified"])
    )
    assert not np.any(np.asarray(result["w2_wages_from_qualified_business"]) < 0)
    np.testing.assert_array_equal(
        np.asarray(result["self_employment_income"])
        + np.asarray(result["sstb_self_employment_income_before_lsr"]),
        original_self_employment,
    )
    np.testing.assert_array_equal(
        result["sstb_w2_wages_from_qualified_business"],
        np.where(is_sstb, result["w2_wages_from_qualified_business"], 0.0),
    )
    np.testing.assert_array_equal(
        result["sstb_unadjusted_basis_qualified_property"],
        np.where(
            is_sstb,
            result["unadjusted_basis_qualified_property"],
            0.0,
        ),
    )


def test_summary_uses_supplied_weights_and_rejects_bad_weights() -> None:
    outputs = simulate_qbi_inputs(
        QbiSimulationInputs.from_puf_arrays(_synthetic_sources()),
        assumptions=load_qbi_simulation_assumptions(QBI_SIMULATION_VERSION),
        qbi_simulation_version=QBI_SIMULATION_VERSION,
    )
    weights = np.arange(1, 65, dtype=np.float64)
    summary = qbi_simulation_summary(outputs, weights=weights)

    business_is_sstb = np.asarray(outputs["business_is_sstb"])
    assert summary["business_is_sstb"]["nonzero_rows"] == 5
    assert summary["business_is_sstb"]["nonzero_share"] == pytest.approx(
        weights[business_is_sstb].sum() / weights.sum()
    )
    with pytest.raises(ValueError, match="align one-for-one"):
        qbi_simulation_summary(outputs, weights=[1.0])


@requires_puf_2024
def test_pinned_artifact_replay_has_archived_hashes_and_expected_distributions() -> (
    None
):
    h5py = pytest.importorskip("h5py")
    assert _PUF_2024_PATH is not None
    keys = (
        "tax_unit_id",
        "household_weight",
        "person_tax_unit_id",
        "self_employment_income",
        "farm_rent_income",
        "rental_income",
        "estate_income",
        "partnership_s_corp_income",
        "non_qualified_dividend_income",
        "w2_wages_from_qualified_business",
    )
    with h5py.File(_PUF_2024_PATH) as artifact:
        arrays = {key: artifact[key][:] for key in keys}

    tax_unit_ids = np.asarray(arrays["tax_unit_id"])
    person_tax_unit_ids = np.asarray(arrays["person_tax_unit_id"])
    assert np.all(tax_unit_ids[:-1] <= tax_unit_ids[1:])
    tax_unit_positions = np.searchsorted(tax_unit_ids, person_tax_unit_ids)
    assert np.all(tax_unit_ids[tax_unit_positions] == person_tax_unit_ids)
    person_weights = np.asarray(arrays["household_weight"])[tax_unit_positions]

    result = with_qbi_simulation_from_puf_arrays(
        arrays,
        qbi_simulation_version=QBI_SIMULATION_VERSION,
    )
    expected_hashes = {
        "business_is_sstb": (
            "4778f17242950bdd9ba1ea2a56b4b31e13c026c077042994718be40357020d5e"
        ),
        "qualified_bdc_income": (
            "0f97dc7dee3547699d800cd7be10aa8eee65d9a27d1f6ab1e63653d02726a2eb"
        ),
        "qualified_reit_and_ptp_income": (
            "e913f2e02ec3f8cd637ecc4d71f89268d611abcc13b0bb47d218d782acf68faf"
        ),
        "unadjusted_basis_qualified_property": (
            "d818169f775b24a34a3c67509ffeee35ecd13063e3b08bed0de850469972964f"
        ),
        "w2_wages_from_qualified_business": (
            "3fa8f57f4bfa80c59703323bc33eafe2322396002397e2b4fb4a1be3eaf008fb"
        ),
    }
    assert {
        column: _sha256(np.asarray(result[column])) for column in expected_hashes
    } == expected_hashes

    summary = qbi_simulation_summary(result, weights=person_weights)
    expected_moments = {
        "business_is_sstb": (32_275, 0.0327537656253, 0.0327537656253),
        "qualified_bdc_income": (8_141, 0.00673979493245, 0.458362362036),
        "qualified_reit_and_ptp_income": (
            59_494,
            0.0490069937076,
            42.60363289,
        ),
        "sstb_self_employment_income_before_lsr": (
            18_938,
            0.0270531721298,
            599.511232106,
        ),
        "sstb_unadjusted_basis_qualified_property": (
            11_561,
            0.00967362760052,
            1_930.42349741,
        ),
        "sstb_w2_wages_from_qualified_business": (
            5_717,
            0.000252356260547,
            405.646584441,
        ),
        "unadjusted_basis_qualified_property": (
            54_301,
            0.0478208552378,
            9_513.1560346,
        ),
        "w2_wages_from_qualified_business": (
            22_516,
            0.000939016471563,
            1_567.19411416,
        ),
    }
    for column, (rows, share, mean) in expected_moments.items():
        assert summary[column]["nonzero_rows"] == rows
        assert summary[column]["nonzero_share"] == pytest.approx(
            share,
            abs=1e-12,
        )
        assert summary[column]["weighted_mean"] == pytest.approx(
            mean,
            rel=1e-10,
        )

    band_misses: list[str] = []
    bands = {
        **qbi_inputs_module._BOOLEAN_SHARE_BANDS,
        **qbi_inputs_module._NUMERIC_NONZERO_SHARE_BANDS,
    }
    for column, (low, high) in bands.items():
        share = float(summary[column]["nonzero_share"])
        if not low <= share <= high:
            band_misses.append(column)
    assert band_misses == ["w2_wages_from_qualified_business"]

    physical_w2 = np.asarray(arrays["w2_wages_from_qualified_business"])
    replay_w2 = np.asarray(result["w2_wages_from_qualified_business"])
    assert np.count_nonzero(physical_w2) == 125_162
    assert person_weights[physical_w2 != 0].sum() / person_weights.sum() == (
        pytest.approx(0.122050921274)
    )
    assert np.sum(physical_w2 * person_weights) / person_weights.sum() == (
        pytest.approx(1_167.93253636)
    )
    assert not np.array_equal(physical_w2, replay_w2)
    assert not np.any((physical_w2 > 0) & (replay_w2 > 0) & (physical_w2 == replay_w2))
    assert np.count_nonzero((physical_w2 == 0) & (replay_w2 == 0)) == 358_607


@requires_puf_2024
def test_v2_reit_ptp_replay_is_within_provisional_published_anchor_band() -> None:
    h5py = pytest.importorskip("h5py")
    assert _PUF_2024_PATH is not None
    keys = (
        "tax_unit_id",
        "household_weight",
        "person_tax_unit_id",
        "self_employment_income",
        "farm_rent_income",
        "rental_income",
        "estate_income",
        "partnership_s_corp_income",
        "non_qualified_dividend_income",
    )
    with h5py.File(_PUF_2024_PATH) as artifact:
        arrays = {key: artifact[key][:] for key in keys}

    tax_unit_ids = np.asarray(arrays["tax_unit_id"])
    person_tax_unit_ids = np.asarray(arrays["person_tax_unit_id"])
    tax_unit_positions = np.searchsorted(tax_unit_ids, person_tax_unit_ids)
    assert np.all(tax_unit_ids[tax_unit_positions] == person_tax_unit_ids)
    person_weights = np.asarray(arrays["household_weight"])[tax_unit_positions]

    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_V2)
    result = with_qbi_simulation_from_puf_arrays(
        arrays,
        qbi_simulation_version=QBI_SIMULATION_V2,
        assumptions=assumptions,
    )
    aggregate = float(
        np.sum(np.asarray(result["qualified_reit_and_ptp_income"]) * person_weights)
    )
    anchor = assumptions.reit_ptp_anchor

    assert anchor.provisional
    assert anchor.published_income_dollars == 21_070_000_000.0
    assert anchor.replay_factor_band == (0.3, 3.0)
    low_factor, high_factor = anchor.replay_factor_band
    assert (
        low_factor * anchor.published_income_dollars
        <= aggregate
        <= high_factor * anchor.published_income_dollars
    )
