"""The stage-1 donor frame that takes every rare output's carriers (microcosm#1276).

The FACT-weighted bootstrap keeps a record in proportion to its weight, so it
drops most of the tape's low-weight, high-income records, where the rare
outputs sit. ``rare_carriers_certain`` keeps the row count, takes every record
carrying a rare output and weights each row by FACT over its inclusion
probability. These tests pin that the frame holds every rare carrier once,
that its weighted incidence is the tape's, that the draw is seeded, and that
the stage refuses a manifest whose weights contradict its design. The stage-1
gates also offset declared predictors' levels in their calibration
(``gate_level_columns``), so the composite records' age band and the other
margins each keep their own rate; the last tests pin how the declared names
map to the encoded columns and that the stage refuses names it cannot map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.spi_income import (
    SPI_DONOR_FRAME_FACT_BOOTSTRAP,
    SPI_DONOR_FRAME_FIT_WEIGHT,
    SPI_DONOR_FRAME_RARE_CARRIERS_CERTAIN,
    SPI_DONOR_RARE_CARRIER_SHARE,
    SPI_STAGE1_CATEGORICAL_PREDICTORS,
    SPI_STAGE1_PREDICTOR_COLUMNS,
    _capped_inclusion,
    _check_gate_level_columns,
    _encode_predictor_frames,
    _encoded_gate_level_columns,
    _rare_carriers_certain_frame,
)
from microcosm.build.uk_runtime.spi_support import SPI_INCOME_QRF_OUTPUT_COLUMNS

RARE = "hmrc_spi_taxable_termination_pay"
COMMON = "hmrc_spi_pay"


def _tape(seed: int = 4, n: int = 20_000) -> pd.DataFrame:
    """A tape whose rare output sits on low-weight, high-pay records."""
    rng = np.random.default_rng(seed)
    high = rng.random(n) < 0.15
    tape = pd.DataFrame(
        {column: np.zeros(n) for column in SPI_INCOME_QRF_OUTPUT_COLUMNS}
    )
    tape["FACT"] = np.where(
        high, rng.uniform(1.0, 10.0, n), rng.uniform(40.0, 120.0, n)
    )
    tape[COMMON] = np.where(
        high, rng.lognormal(11.5, 0.4, n), rng.lognormal(10.0, 0.5, n)
    )
    carrier = high & (rng.random(n) < 0.05)
    tape[RARE] = np.where(carrier, rng.lognormal(10.0, 1.0, n), 0.0)
    return tape


def test_capped_inclusion_sums_to_the_size_and_never_exceeds_one() -> None:
    weights = np.array([1.0, 1.0, 2.0, 50.0, 400.0])
    inclusion = _capped_inclusion(weights, 3)
    np.testing.assert_allclose(inclusion.sum(), 3.0)
    assert inclusion.max() == 1.0
    assert (inclusion[-2:] == 1.0).all()
    np.testing.assert_allclose(inclusion[2] / inclusion[0], 2.0)


def test_the_frame_holds_every_rare_carrier_once_at_the_tape_incidence() -> None:
    tape = _tape()
    size = 4_000
    carriers = int((tape[RARE] != 0).sum())
    assert 0 < carriers < SPI_DONOR_RARE_CARRIER_SHARE * size
    frame, weights, receipt = _rare_carriers_certain_frame(tape, size=size, seed=9)

    assert len(frame) == size == receipt["rows"]
    assert receipt["rare_outputs"] == [RARE]
    assert receipt["rows_carrying_a_rare_output"] == carriers
    # Every carrier is in the frame, once, at its own FACT.
    in_frame = frame[RARE] != 0
    assert int(in_frame.sum()) == carriers
    np.testing.assert_allclose(
        np.sort(weights[in_frame.to_numpy()]),
        np.sort(tape.loc[tape[RARE] != 0, "FACT"].to_numpy()),
    )
    # No record repeats: the pay values are continuous, so rows are distinct.
    assert frame[COMMON].nunique() == size
    # The weights carry the tape's population and the rare output's incidence.
    fact = tape["FACT"].to_numpy()
    np.testing.assert_allclose(weights.sum(), fact.sum())
    np.testing.assert_allclose(
        (weights * frame[RARE].to_numpy()).sum(), (fact * tape[RARE].to_numpy()).sum()
    )
    np.testing.assert_allclose(
        weights[in_frame.to_numpy()].sum(), fact[(tape[RARE] != 0).to_numpy()].sum()
    )
    # A common output is estimated, not exact: within a few percent here.
    estimate = (weights * frame[COMMON].to_numpy()).sum()
    assert abs(estimate / (fact * tape[COMMON].to_numpy()).sum() - 1.0) < 0.05


def test_the_frame_draw_is_seeded() -> None:
    tape = _tape()
    first, first_weights, _ = _rare_carriers_certain_frame(tape, size=4_000, seed=9)
    again, again_weights, _ = _rare_carriers_certain_frame(tape, size=4_000, seed=9)
    other, _, _ = _rare_carriers_certain_frame(tape, size=4_000, seed=10)
    pd.testing.assert_frame_equal(first, again)
    np.testing.assert_array_equal(first_weights, again_weights)
    assert not first[COMMON].equals(other[COMMON])


def test_a_tape_no_larger_than_the_frame_is_the_frame() -> None:
    tape = _tape(n=300)
    frame, weights, receipt = _rare_carriers_certain_frame(tape, size=4_000, seed=9)
    assert len(frame) == 300 and receipt["whole_tape"] is True
    np.testing.assert_array_equal(weights, tape["FACT"].to_numpy())


def test_each_design_names_its_fit_weight() -> None:
    assert SPI_DONOR_FRAME_FIT_WEIGHT == {
        SPI_DONOR_FRAME_FACT_BOOTSTRAP: "uniform",
        SPI_DONOR_FRAME_RARE_CARRIERS_CERTAIN: "inverse_inclusion",
    }


def test_the_stage_refuses_a_fit_weight_that_contradicts_its_design() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime import spi_spine

    stage = load_country_spec("uk").sources.stage_map()["hmrc_spi_income_spine"]
    assert spi_spine.declared_donor_frame_design(stage) == (
        SPI_DONOR_FRAME_RARE_CARRIERS_CERTAIN
    )
    spi_spine._assert_income_stage_parameters(
        stage, seed=42, qrf_estimators=100, donor_sample_size=100_000
    )
    stage1 = spi_spine._operation(stage, "fit_weighted_qrf_stage1")
    drifted = {**stage1.parameters, "post_sample_fit_weight": "uniform"}
    with pytest.raises(ValueError, match="must fit at 'inverse_inclusion' weight"):
        spi_spine._assert_income_stage_parameters(
            _with_stage1_parameters(stage, drifted),
            seed=42,
            qrf_estimators=100,
            donor_sample_size=100_000,
        )


def _with_stage1_parameters(stage, parameters):
    from dataclasses import replace

    operations = tuple(
        replace(operation, parameters=parameters)
        if operation.kind == "fit_weighted_qrf_stage1"
        else operation
        for operation in stage.operations
    )
    return replace(stage, operations=operations)


def _encoded_stage1_predictors() -> pd.DataFrame:
    """Stage-1 predictors one-hot encoded as the stage encodes them."""
    frame = pd.DataFrame(
        {
            "spi_age_band": [-1.0, 1.0, 2.0, 1.0],
            "gender": ["MALE", "FEMALE", "MALE", "FEMALE"],
            "region": ["LONDON", "WALES", "LONDON", "WALES"],
            "main_source": ["pay", "trade", "pay", "pension"],
            "has_pay": [1.0, 0.0, 1.0, 0.0],
            "has_trade": [0.0, 1.0, 0.0, 0.0],
            "at_or_over_spa": [0.0, 0.0, 1.0, 1.0],
        }
    )
    (encoded,) = _encode_predictor_frames(
        (frame[list(SPI_STAGE1_PREDICTOR_COLUMNS)],),
        categorical=SPI_STAGE1_CATEGORICAL_PREDICTORS,
    )
    return encoded


def test_gate_level_columns_take_every_one_hot_level_of_a_categorical() -> None:
    encoded = _encoded_stage1_predictors()
    levels = _encoded_gate_level_columns(
        ("spi_age_band", "main_source", "has_pay", "has_trade"), encoded.columns
    )
    assert levels == (
        "spi_age_band_-1.0",
        "spi_age_band_1.0",
        "spi_age_band_2.0",
        "main_source_pay",
        "main_source_pension",
        "main_source_trade",
        "has_pay",
        "has_trade",
    )
    assert set(levels) <= set(encoded.columns)
    assert _encoded_gate_level_columns((), encoded.columns) == ()


def test_gate_level_columns_must_be_distinct_encodable_stage1_predictors() -> None:
    with pytest.raises(ValueError, match="not stage-1 predictors"):
        _check_gate_level_columns(("age",))
    with pytest.raises(ValueError, match="repeats a column"):
        _check_gate_level_columns(("has_pay", "has_pay"))
    with pytest.raises(ValueError, match="has no encoded predictor"):
        _encoded_gate_level_columns(("has_pay",), ["gender_MALE"])


def test_the_stage_refuses_gate_level_columns_outside_its_predictors() -> None:
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime import spi_spine

    stage = load_country_spec("uk").sources.stage_map()["hmrc_spi_income_spine"]
    stage1 = spi_spine._operation(stage, "fit_weighted_qrf_stage1")
    drifted = {**stage1.parameters, "gate_level_columns": ["spi_age_band", "age"]}
    with pytest.raises(ValueError, match="gate_level_columns must be stage-1"):
        spi_spine._assert_income_stage_parameters(
            _with_stage1_parameters(stage, drifted),
            seed=42,
            qrf_estimators=100,
            donor_sample_size=100_000,
        )
