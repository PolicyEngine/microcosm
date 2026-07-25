"""Versioned Populace Section 199A simulation tests."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from populace.build.us_runtime import qbi_inputs as qbi_inputs_module
from populace.build.us_runtime.qbi_inputs import US_QBI_OUTPUT_COLUMNS
from populace.build.us_runtime.qbi_simulation import (
    QBI_SIMULATION_SOURCE_NAMES,
    QBI_SIMULATION_VERSION,
    QbiSimulationInputs,
    load_qbi_simulation_assumptions,
    qbi_simulation_summary,
    simulate_qbi_inputs,
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


def test_version_and_random_stream_order_are_strict() -> None:
    assumptions = load_qbi_simulation_assumptions(QBI_SIMULATION_VERSION)
    inputs = QbiSimulationInputs.from_puf_arrays(_synthetic_sources())

    with pytest.raises(ValueError, match="Unsupported qbi_simulation_version"):
        load_qbi_simulation_assumptions(2)
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
