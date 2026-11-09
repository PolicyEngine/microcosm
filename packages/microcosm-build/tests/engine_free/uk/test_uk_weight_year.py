"""The year a UK release's household weights are written at (microcosm#796 A19)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import hmrc_uprating, weight_year
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.weight_year import (
    UK_ENGINE_YEAR_MASS_REASON,
    UK_WEIGHT_PROJECTION_PARAMETER,
    UK_WEIGHT_PROJECTION_PINS_BLOCK,
    UK_WEIGHT_PROJECTION_VARIABLE,
    UK_WEIGHT_YEAR_MASS_REASON,
    uk_dataset_year_frame,
    uk_engine_year_frame,
    uk_solve_year_frame,
    uk_weight_projection_factor,
    uk_weight_projection_growth,
    uk_weights_are_at_dataset_year,
)
from microcosm.frame import WeightKind

SOLVED = np.asarray([1_000.0, 2_500.0, 400.0])


def _frame(*, kind: WeightKind = WeightKind.CALIBRATED, period: str = "2024"):
    ids = np.asarray([1, 2, 3], dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {"person_id": ids, "person_benunit_id": ids, "person_household_id": ids}
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame({"household_id": ids}),
        time_period=period,
        weight_kind=kind,
        household_weights=SOLVED,
    )


def _growth(year: int) -> float:
    pins = hmrc_uprating.engine_parameter_pins()
    block = pins[UK_WEIGHT_PROJECTION_PINS_BLOCK][UK_WEIGHT_PROJECTION_VARIABLE]
    return float(block["growth"][f"{year}-01-01"])


def test_the_pins_carry_the_engine_growth_of_household_weight() -> None:
    pins = hmrc_uprating.engine_parameter_pins()
    block = pins[UK_WEIGHT_PROJECTION_PINS_BLOCK]
    assert list(block) == [UK_WEIGHT_PROJECTION_VARIABLE]
    entry = block[UK_WEIGHT_PROJECTION_VARIABLE]
    assert entry["parameter"] == UK_WEIGHT_PROJECTION_PARAMETER
    # One growth rate per pinned instant, the same instants as the indices.
    assert list(entry["growth"]) == pins["instants"]
    assert uk_weight_projection_growth(2025) == _growth(2025)


def test_the_factor_compounds_the_growth_after_the_dataset_year() -> None:
    assert uk_weight_projection_factor(2024, 2024) == 1.0
    assert uk_weight_projection_factor(2024, 2025) == 1.0 + _growth(2025)
    assert uk_weight_projection_factor(2024, 2026) == pytest.approx(
        (1.0 + _growth(2025)) * (1.0 + _growth(2026)), rel=0, abs=0
    )
    # The dataset's own year takes no growth: only the years after it do.
    assert uk_weight_projection_factor(2025, 2026) == 1.0 + _growth(2026)


def test_the_engine_does_not_project_backwards() -> None:
    with pytest.raises(ValueError, match="does not project a 2025 dataset back"):
        uk_weight_projection_factor(2025, 2024)


def test_a_year_the_pins_do_not_cover_refuses() -> None:
    with pytest.raises(ValueError, match="re-run"):
        uk_weight_projection_growth(2099)


def test_pins_without_the_block_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    pins = dict(hmrc_uprating.engine_parameter_pins())
    pins.pop(UK_WEIGHT_PROJECTION_PINS_BLOCK)
    monkeypatch.setattr(weight_year, "engine_parameter_pins", lambda: pins)

    with pytest.raises(ValueError, match="does not pin"):
        uk_weight_projection_growth(2025)


def test_release_weights_are_the_solved_weights_over_the_engine_factor() -> None:
    frame = _frame()
    factor = uk_weight_projection_factor(2024, 2025)

    released = uk_dataset_year_frame(frame, solve_year=2025)

    weights = released.weights_for("household")
    assert weights.kind is WeightKind.CALIBRATED
    np.testing.assert_array_equal(weights.values, SOLVED / factor)
    record = released.mass_log[-1]
    assert record.entity == "household"
    assert record.reason == UK_WEIGHT_YEAR_MASS_REASON
    assert record.declared_factor == 1.0 / factor
    assert record.old_total == SOLVED.sum()
    assert uk_weights_are_at_dataset_year(released)
    # The solved frame is not touched, and it carries no such record.
    np.testing.assert_array_equal(frame.weights_for("household").values, SOLVED)
    assert not uk_weights_are_at_dataset_year(frame)
    # The solve's evidence finds its own record by the word "calibration" on
    # the frame before this one; the division must never be mistaken for it.
    assert "calibration" not in UK_WEIGHT_YEAR_MASS_REASON
    assert "calibration" not in UK_ENGINE_YEAR_MASS_REASON


def test_the_division_is_applied_once() -> None:
    released = uk_dataset_year_frame(_frame(), solve_year=2025)

    again = uk_dataset_year_frame(released, solve_year=2025)

    assert again is released


def test_only_solved_weights_are_divided() -> None:
    with pytest.raises(ValueError, match="Only solved household weights"):
        uk_dataset_year_frame(_frame(kind=WeightKind.DESIGN), solve_year=2025)


def test_the_engine_projection_of_a_release_gives_back_the_solved_weights() -> None:
    released = uk_dataset_year_frame(_frame(), solve_year=2025)

    at_solve_year = uk_engine_year_frame(released, simulation_year=2025)

    np.testing.assert_allclose(
        at_solve_year.weights_for("household").values, SOLVED, rtol=1e-15, atol=0
    )
    assert at_solve_year.mass_log[-1].reason == UK_ENGINE_YEAR_MASS_REASON
    # At the dataset's own year the engine applies no growth.
    assert uk_engine_year_frame(released, simulation_year=2024) is released


def test_a_file_written_as_solved_reads_high_at_the_solve_year() -> None:
    # The defect this fixes: weights solved at 2025 and stored on a 2024
    # dataset come back multiplied by the population growth of 2025.
    as_solved = _frame()

    seen = uk_engine_year_frame(as_solved, simulation_year=2025)

    np.testing.assert_array_equal(
        seen.weights_for("household").values,
        SOLVED * uk_weight_projection_factor(2024, 2025),
    )


def test_a_release_goes_back_to_its_solved_weights_for_the_gates() -> None:
    released = uk_dataset_year_frame(_frame(), solve_year=2025)

    solved = uk_solve_year_frame(released, solve_year=2025)

    np.testing.assert_allclose(
        solved.weights_for("household").values, SOLVED, rtol=1e-15, atol=0
    )
    # A file written as solved already carries them.
    as_solved = _frame()
    assert uk_solve_year_frame(as_solved, solve_year=2025) is as_solved


def test_a_release_divided_for_another_year_refuses() -> None:
    released = uk_dataset_year_frame(_frame(), solve_year=2026)

    with pytest.raises(ValueError, match="would not return the solved weights"):
        uk_solve_year_frame(released, solve_year=2025)


def test_the_division_survives_the_release_file(tmp_path) -> None:
    released = uk_dataset_year_frame(_frame(), solve_year=2025)
    path = tmp_path / "release.h5"

    write_uk_national_frame(released, path)
    loaded, _provenance = load_uk_national_frame(path)

    assert uk_weights_are_at_dataset_year(loaded)
    np.testing.assert_array_equal(
        loaded.weights_for("household").values,
        released.weights_for("household").values,
    )
    assert loaded.mass_log == released.mass_log
