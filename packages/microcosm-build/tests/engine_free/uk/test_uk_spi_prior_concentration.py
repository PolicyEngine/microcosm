"""The over-£100,000 young-child cell's concentration figures (Vahid's #1338 item 4)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.diagnose_uk_spi_prior_concentration import (
    LARGEST_SOURCES,
    concentration,
    kish_effective_size,
)


def test_kish_effective_size_counts_equal_weights_as_units() -> None:
    assert kish_effective_size(np.ones(40)) == pytest.approx(40.0)
    # One heavy unit shrinks the effective size; zero weights do not count.
    assert kish_effective_size(np.array([9.0, 1.0, 0.0])) == pytest.approx(100 / 82)
    assert kish_effective_size(np.zeros(3)) == 0.0


def test_concentration_pools_copies_with_their_source_household() -> None:
    # Twelve source households; the first is one FRS row and three SPI copies.
    weights = np.array([5.0, 5.0, 5.0, 5.0, *np.ones(11)])
    sources = np.array([1, 1, 1, 1, *range(2, 13)])
    spi_copy = np.array([False, True, True, True, *[False] * 11])

    figures = concentration(weights, sources, spi_copy)

    assert figures["weighted_households"] == pytest.approx(31.0)
    assert figures["kish_effective_households"] == pytest.approx(31**2 / 111)
    # The source household of joint weight 20 counts once.
    assert figures["kish_effective_source_households"] == pytest.approx(31**2 / 411)
    # The ten largest sources: the joint 20 and nine of the eleven at 1.
    key = f"largest_{LARGEST_SOURCES}_source_households_weight_share"
    assert figures[key] == pytest.approx(29 / 31)
    assert figures["spi_copy_weight_share"] == pytest.approx(15 / 31)
    assert figures["source_households"] == 10


def test_concentration_withholds_a_cell_on_one_or_two_source_households() -> None:
    figures = concentration(
        np.array([3.0, 4.0]), np.array([7, 8]), np.array([False, True])
    )
    assert figures == {"source_households": "<3", "withheld": True}
    assert concentration(np.array([]), np.array([]), np.array([], dtype=bool)) == {
        "source_households": 0,
        "withheld": True,
    }
