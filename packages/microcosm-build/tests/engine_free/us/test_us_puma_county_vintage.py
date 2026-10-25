"""PUMA planning-region draws preserve the assigned legacy county and PUMA."""

import numpy as np
import pytest

from microcosm.build.us_runtime.county_vintage import county_fips_2023_from_tracts
from microcosm.build.us_runtime.puma_county_vintage import (
    county_fips_2023_for_puma_assignment,
)


def _overlap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Bridgeport and Shelton share Fairfield County but have different regions.
    # Stamford is in a separate PUMA here, so its large population cannot leak
    # into the first pair's conditional draw.
    return (
        np.asarray([900100, 900100, 900200]),
        np.asarray([9001070100, 9001110100, 9001020101]),
        np.asarray([1.0, 3.0, 1000.0]),
    )


def test_stored_tract_resolves_directly_without_random_draws() -> None:
    tracts = np.asarray([9001070100, 9001110100, 6037000100])
    rng = np.random.default_rng(19)
    untouched = np.random.default_rng(19)

    actual = county_fips_2023_for_puma_assignment(
        np.asarray([900100, 900100, 600100]),
        tracts // 1_000_000,
        tract_values=tracts,
        tract_overlap=_overlap(),
        rng=rng,
    )

    np.testing.assert_array_equal(actual, county_fips_2023_from_tracts(tracts))
    assert rng.random() == untouched.random()


def test_non_ct_identity_consumes_no_random_draws() -> None:
    rng = np.random.default_rng(42)
    untouched = np.random.default_rng(42)

    actual = county_fips_2023_for_puma_assignment(
        np.asarray([100100, 600100, 3600100]),
        np.asarray([1001, 6037, 36061]),
        tract_values=None,
        tract_overlap=_overlap(),
        rng=rng,
    )

    assert actual.tolist() == ["01001", "06037", "36061"]
    assert rng.random() == untouched.random()


def test_ct_conditional_draw_respects_pair_population_and_is_reproducible() -> None:
    puma = np.asarray([900100] * 20_000 + [900200] * 100 + [600100])
    county = np.asarray([9001] * 20_100 + [6037])
    original_county = county.copy()

    def draw() -> np.ndarray:
        return county_fips_2023_for_puma_assignment(
            puma,
            county,
            tract_values=None,
            tract_overlap=_overlap(),
            rng=np.random.default_rng(24),
        )

    actual = draw()

    np.testing.assert_array_equal(actual, draw())
    np.testing.assert_array_equal(county, original_county)
    assert len(actual) == len(puma)
    assert set(actual[:20_000]) == {"09120", "09140"}
    assert 0.72 < np.mean(actual[:20_000] == "09140") < 0.78
    assert set(actual[20_000:20_100]) == {"09190"}
    assert actual[-1] == "06037"
    assert all(isinstance(value, str) and len(value) == 5 for value in actual)


def test_missing_conditional_pair_fails_closed() -> None:
    with pytest.raises(ValueError, match="0900300.*09001"):
        county_fips_2023_for_puma_assignment(
            np.asarray([900300]),
            np.asarray([9001]),
            tract_values=None,
            tract_overlap=_overlap(),
            rng=np.random.default_rng(0),
        )
