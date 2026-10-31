"""Receipt counts of licensed records are banded at source (microcosm#1164)."""

from __future__ import annotations

import re

import pytest

from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    distinct_count,
    extreme_count,
    floored_share,
    rounded_count,
    small_cell_share,
    support_band,
)
from test_support.paths import paths_for


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        # The floor of three (María's ruling of 2026-10-29): one or two are
        # withheld, three to nine read "3-9" (the review of microcosm#1422),
        # then the rounding starts.
        (1, "<3"),
        (2, "<3"),
        (3, "3-9"),
        (9, "3-9"),
        (10, 10),
        (14, 10),
        (25, 30),
        (994, 990),
        (995, 1000),
        (4321, 4300),
        (4350, 4400),
    ],
)
def test_rounded_count(count, expected) -> None:
    assert rounded_count(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (2, "<3"),
        (3, "3-9"),
        (9, "3-9"),
        (10, "10-49"),
        (49, "10-49"),
        (50, "50-99"),
        (99, "50-99"),
        (100, 100),
        (230, 200),
        (250, 300),
    ],
)
def test_extreme_count(count, expected) -> None:
    assert extreme_count(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (1, "<3"),
        (2, "<3"),
        (3, "3-9"),
        (9, "3-9"),
        (10, "10-29"),
        (29, "10-29"),
        (30, "30+"),
    ],
)
def test_support_band(count, expected) -> None:
    # An empty cell reads 0, as the other counts here, never "<3" beside a
    # zero weight.
    assert support_band(count) == expected


def test_no_uk_output_still_withholds_at_ten() -> None:
    """Every UK receipt, gate note and tool withholds through
    ``UK_DISCLOSURE_FLOOR``. One that still writes the old "<10" label stays on
    the floor of ten without anyone noticing, as the SPI pay-floor receipt of
    microcosm#1426 would if it landed second (the review of microcosm#1422)."""

    root = paths_for("microcosm-build").repository
    build = paths_for("microcosm-build").package / "src" / "microcosm" / "build"
    label = re.compile(r"<10(?![\d%])")
    writers = sorted(
        path.relative_to(root).as_posix()
        for path in (
            *(build / "uk_runtime").rglob("*.py"),
            *(build / "uk").glob("*.py"),
            *(build / "uk" / "spec").glob("*.yaml"),
            build / "uk" / "gates.json",
            *(root / "tools").rglob("*.py"),
            *(paths_for("microcosm-data").package / "src").rglob("*.py"),
        )
        if label.search(path.read_text(encoding="utf-8"))
    )
    assert writers == []


#: Floors of ten that are not the disclosure floor, each with the text that
#: names it on its line: statistical floors (too few rows to measure what the
#: figure measures), and lines that describe the old floor as history.
_KEPT_TENS = {
    "uk_runtime/frs_annual_income.py": (
        "PAY_CONCEPT_RATIO_MINIMUM_SUPPORT = 10",
        "support band is under ten",
    ),
    "uk_runtime/stage_health.py": ("pay concept ratio on fewer than ten",),
    "uk_runtime/spi_rank_transport.py": ("SPI_TRANSPORT_GROUP_MINIMUM_ROWS = 10",),
    "uk_runtime/private_rent_localisation.py": ("MIN_SAMPLED_REGION_RENTERS = 10",),
    "uk_runtime/frs_education.py": ("AGE_LEFT_FULL_TIME_EDUCATION_MIN = 10",),
    "uk_runtime/frs_spine.py": ("fewer than ten code-6 amounts",),
    "uk_runtime/orrery_contract.py": (
        "no longer marks the rows resting on fewer than ten",
    ),
    "uk_runtime/support_floor.py": ("rests on fewer than ten records is no reason",),
    "uk/spec/sources.yaml": ("over at least 10 rows",),
    "tools/measure_uk_spi_half_coherence.py": ("MIN_CELL = 10",),
    "tools/uk_private_rent_receipt.py": (
        "MIN_GROUP_CELLS = 10",
        "fewer than ten published rows",
    ),
}


def test_no_uk_floor_of_ten_hides_outside_the_label() -> None:
    """The "<10" label is only one way to keep a floor of ten: a comparison
    with ten, a constant or a default of ten, or prose saying "fewer than ten"
    is another (the review of microcosm#1422). Each that is left is a
    statistical floor or history, named in ``_KEPT_TENS``."""

    root = paths_for("microcosm-build").repository
    build = paths_for("microcosm-build").package / "src" / "microcosm" / "build"
    floors = re.compile(
        r"\b\w*(?:support|sources?|count|households|donors|carriers|rows|cells)"
        r"\w*\)?\s*<\s*10\b(?!\.\d)"
        r"|^\s*_?[A-Z0-9_]*(?:MIN|FLOOR|SDC|SMALL)[A-Z0-9_]*\s*(?::[^=\n]+)?=\s*10\b"
        r"|\bminimum_\w*\s*:\s*int\s*=\s*10\b"
        r"|\b(?:fewer than|under|below|less than) (?:ten|10 (?:rows|records|households"
        r"|source households|sources|donors))\b|\bone to nine\b"
        r"|\bat least 10 (?:rows|records|households|sources|donors)\b"
        r'|"minimum_\w*":\s*10\b',
        re.IGNORECASE | re.MULTILINE,
    )
    paths = (
        *(build / "uk_runtime").rglob("*.py"),
        *(build / "uk").glob("*.py"),
        *(build / "uk" / "spec").glob("*.yaml"),
        build / "uk" / "gates.json",
        *(root / "tools").rglob("*.py"),
        *(paths_for("microcosm-data").package / "src").rglob("*.py"),
    )
    found = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(root).as_posix()
        kept = next(
            (
                needles
                for suffix, needles in _KEPT_TENS.items()
                if name.endswith(suffix)
            ),
            (),
        )
        for match in floors.finditer(text):
            line = text[
                text.rfind("\n", 0, match.start()) + 1 : text.find("\n", match.end())
            ]
            if not any(needle in line for needle in kept):
                found.append(f"{name}: {line.strip()[:120]}")
    assert found == []


def test_counts_refuse_a_negative_count() -> None:
    with pytest.raises(ValueError, match="negative"):
        rounded_count(-1)
    with pytest.raises(ValueError, match="negative"):
        extreme_count(-1)
    with pytest.raises(ValueError, match="negative"):
        support_band(-1)


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.0, 0.0), (0.987, 0.98), (0.98, 0.98), (0.999, 0.99), (1.0, 1.0)],
)
def test_floored_share(share, expected) -> None:
    assert floored_share(share) == expected


def test_floored_share_refuses_a_share_outside_zero_one() -> None:
    with pytest.raises(ValueError, match="share"):
        floored_share(1.2)


def test_small_cell_share() -> None:
    assert small_cell_share(0.0, 0) == 0.0
    assert small_cell_share(0.01, 2) == "<3"
    assert small_cell_share(0.01, 3) == pytest.approx(0.01)
    assert small_cell_share(0.25, 40) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="negative"):
        small_cell_share(0.1, -1)


def test_distinct_carriers_count_keyed_rows_of_positive_weight() -> None:
    """The receipts' one carrier rule (the third review of #1305)."""

    keys = ["a", "a", "b", "", None, "c"]
    assert distinct_carriers(keys) == 3
    assert distinct_carriers(keys, [1.0, 0.0, 2.0, 1.0, 1.0, 0.0]) == 2
    with pytest.raises(ValueError, match="align"):
        distinct_carriers(keys, [1.0])


def test_a_key_counts_once_however_spelled_and_a_missing_key_never() -> None:
    """The fourth review of #1305: NaN counted as a carrier, and 1 and 1.0 as two."""

    import numpy as np
    import pandas as pd

    assert distinct_carriers([1, 1.0, np.int64(1), np.float64(1.0), "1"]) == 1
    assert distinct_carriers([np.nan, None, "", "  ", pd.NA, pd.NaT]) == 0
    assert distinct_carriers(["2024:1", "2024:2", 2.5]) == 3


def test_a_cell_is_counted_by_the_same_rule() -> None:
    """#1308's stage receipts count a cell's sources as the receipts do."""

    import numpy as np

    ids = np.array([1, 2, 2, 3, 4])
    cell = np.array([True, True, True, True, False])
    assert distinct_count(ids, cell) == 3
    assert distinct_count(ids, cell, np.array([1.0, 0.0, 0.0, 2.0, 1.0])) == 2
