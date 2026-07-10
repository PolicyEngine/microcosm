from __future__ import annotations

import numpy as np
import pytest

from populace.build.uk_runtime import (
    MEASURE_TO_COLUMN,
    exposure_for_major_group,
    exposure_for_soc,
    load_ai_exposure_table,
    load_major_group_ai_exposure_table,
)


def test_unit_group_table_covers_all_soc2020_unit_groups() -> None:
    table = load_ai_exposure_table()

    assert len(table) == 412
    assert table.index.str.fullmatch(r"\d{4}").all()
    assert set(table["mapping_quality"]) <= {
        "direct",
        "chained",
        "imputed-from-parent",
    }
    for column in MEASURE_TO_COLUMN.values():
        assert table[column].notna().all()


def test_known_codes_return_scores_in_expected_ranges() -> None:
    theta = exposure_for_soc(["1111", "7113", "2211"], measure="complementarity")
    assert ((theta > 0.0) & (theta < 1.0)).all()
    # Medical practitioners are more AI-complementary than telephone
    # salespersons (Pizzinelli et al. 2023's lawyers/telemarketers contrast).
    assert theta[2] > theta[1]

    c_aioe = exposure_for_soc(["4112", "9267"], measure="c_aioe")
    assert np.isfinite(c_aioe).all()
    assert (np.abs(c_aioe) < 3.0).all()
    # Administrative occupations are more exposed than elementary ones.
    assert c_aioe[0] > c_aioe[1]

    eloundou = exposure_for_soc(["4112"], measure="eloundou_beta")
    assert 0.0 <= eloundou[0] <= 1.0

    # DSIT Annex 1 publishes telephone salespersons (SOC 7113, unchanged
    # between SOC 2010 and SOC 2020) at 1.101573 (AI) and 1.966413 (LLM).
    assert exposure_for_soc(["7113"], measure="dsit_aioe")[0] == pytest.approx(
        1.101573, abs=1e-4
    )
    assert exposure_for_soc(["7113"], measure="dsit_llm")[0] == pytest.approx(
        1.966413, abs=1e-4
    )


def test_integer_codes_match_string_codes() -> None:
    assert exposure_for_soc([4112], measure="felten_aioe")[0] == pytest.approx(
        exposure_for_soc(["4112"], measure="felten_aioe")[0]
    )


def test_parent_code_fallback() -> None:
    table = load_ai_exposure_table()
    expected_411 = table.loc[table.index.str.startswith("411"), "c_aioe"].mean()

    # A 3-digit code resolves to the mean of its unit groups.
    assert exposure_for_soc(["411"])[0] == pytest.approx(expected_411)
    # An unknown 4-digit code falls back to its 3-digit parent.
    assert "4119" not in table.index
    assert exposure_for_soc(["4119"])[0] == pytest.approx(expected_411)

    # A code with no 3-digit parent falls back to the 2-digit parent.
    expected_41 = table.loc[table.index.str.startswith("41"), "c_aioe"].mean()
    assert not table.index.str.startswith("419").any()
    assert exposure_for_soc(["4190"])[0] == pytest.approx(expected_41)


def test_unknown_code_returns_nan_with_warning() -> None:
    with pytest.warns(UserWarning, match="0000"):
        result = exposure_for_soc(["0000", "1111"])

    assert np.isnan(result[0])
    assert np.isfinite(result[1])

    with pytest.warns(UserWarning):
        assert np.isnan(exposure_for_soc(["not-a-code"])[0]).all()


def test_invalid_measure_raises() -> None:
    with pytest.raises(ValueError, match="Unknown measure"):
        exposure_for_soc(["1111"], measure="webb_2020")


def test_major_group_lookup() -> None:
    table = load_major_group_ai_exposure_table()
    assert list(table.index) == [str(group) for group in range(1, 10)]
    assert (table["weighting"] == "ashe_2025_table14_jobs").all()

    scores = exposure_for_major_group([1, 2, 9], measure="c_aioe")
    assert np.isfinite(scores).all()
    # Professional occupations are more exposed than elementary occupations.
    assert scores[1] > scores[2]

    theta = exposure_for_major_group(["1", "9"], measure="complementarity")
    assert ((theta > 0.0) & (theta < 1.0)).all()
    # Managers have higher AI complementarity than elementary occupations.
    assert theta[0] > theta[1]

    with pytest.warns(UserWarning, match="0"):
        missing = exposure_for_major_group([0])
    assert np.isnan(missing[0])


def test_major_group_accepts_frs_thousands_coding() -> None:
    # FRS adult.tab codes SOC 2020 major groups as 1000-9000.
    frs_codes = [group * 1000 for group in range(1, 10)]
    plain_codes = list(range(1, 10))

    for measure in ("c_aioe", "complementarity", "felten_aioe"):
        np.testing.assert_allclose(
            exposure_for_major_group(frs_codes, measure=measure),
            exposure_for_major_group(plain_codes, measure=measure),
        )

    # String FRS coding works too, and non-multiples of 1000 stay unknown.
    np.testing.assert_allclose(
        exposure_for_major_group(["4000"]), exposure_for_major_group(["4"])
    )
    with pytest.warns(UserWarning, match="4100"):
        assert np.isnan(exposure_for_major_group(["4100"])[0])
