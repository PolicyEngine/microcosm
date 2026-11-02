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


#: Every ten the guard finds that is not a floor of ten on survey records:
#: statistical floors (too few rows to measure what a figure measures), band
#: and rounding edges, history, facts that hold at the floor of three, and
#: tens that count something else. Each entry is the phrase containing a match,
#: its words separated by any whitespace, and how many matches it covers in its
#: file, so a new match fails even where an existing phrase would cover it, and
#: so does an entry that matches nothing.
_KEPT_TENS: dict[str, tuple[tuple[str, int], ...]] = {
    "uk_runtime/frs_annual_income.py": (
        ("PAY_CONCEPT_RATIO_MINIMUM_SUPPORT = 10", 1),
        ("holds one past them on fewer than ten households", 1),
        ("a ratio whose support band is under ten", 1),
        ("On one # to nine households a ratio past them is noise", 1),
    ),
    "uk_runtime/stage_health.py": (
        ("So is a pay concept ratio on fewer than ten households", 1),
        ("a count under ten prints", 1),
        ("if value >= 10:", 1),
    ),
    "uk_runtime/receipt_counts.py": (
        ("if value < 10:", 3),
        ("goes to HMRC at least ten working days", 1),
    ),
    "uk_runtime/target_weights.py": (("seats under #: ten homes over 2 million", 1),),
    "uk_runtime/spi_rank_transport.py": (("SPI_TRANSPORT_GROUP_MINIMUM_ROWS = 10", 1),),
    "uk_runtime/private_rent_localisation.py": (
        ("MIN_SAMPLED_REGION_RENTERS = 10", 1),
    ),
    "uk_runtime/frs_education.py": (("AGE_LEFT_FULL_TIME_EDUCATION_MIN = 10", 1),),
    "uk_runtime/firm_generation.py": (("if employment <= 9:", 2),),
    "uk_runtime/frs_spine.py": (
        ("fewer than ten code-6 amounts", 1),
        ("fewer than 10 awards", 1),
    ),
    "uk_runtime/orrery_contract.py": (("rows resting on fewer than ten", 1),),
    "uk_runtime/support_floor.py": (
        ("rests on fewer than ten records is no reason", 1),
    ),
    "uk_runtime/national_reconciliation.py": (("len(moved) > 10", 1),),
    "uk_runtime/size_evaluation.py": (("(values > 10)", 1),),
    "uk_runtime/spi_dependency.py": (("goes to HMRC at least ten", 2),),
    "uk/spec/sources.yaml": (("over at least 10 rows (a statistical floor", 1),),
    "uk/calibration_measure_exclusions.json": (
        ("a cell that rests on fewer than 10 records is no reason", 1),
        (
            "rows under 10 source households are withheld, as the BRMA rows' are "
            "(under three",
            1,
        ),
        ("fewer than 10 source households carry them (fewer than three", 1),
        ("fewer than ten source households carry Plan 5 repayments", 1),
    ),
    "uk/target_reference_signed_exclusions.json": (
        ("expected to rest on fewer than 10 records", 2),
    ),
    "uk/local_small_cell_rule.json": (
        ("expected to rest on fewer than 10 records", 1),
    ),
    "uk/uk_population_targets.json": (
        ("Fewer than 10 FRS source households carry that difference", 2),
    ),
    "tools/uk_private_rent_receipt.py": (
        ("MIN_GROUP_CELLS = 10", 1),
        ("summarised only over at least ten cells", 1),
        ("A group of fewer than ten published rows", 1),
    ),
    "tools/measure_uk_spi_half_coherence.py": (("MIN_CELL = 10", 1),),
    "tools/measure_uk_weighted_integrity_baselines.py": (
        ("goes to HMRC at least ten working days", 1),
    ),
    "tools/graph_uk_spine_fixture.py": (("4 <= band <= 9", 1),),
}
#: The hand-written UK JSON files; the generated ones repeat what these say.
_UK_HAND_WRITTEN_JSON = (
    "gates.json",
    "calibration_measure_exclusions.json",
    "target_reference_signed_exclusions.json",
    "local_small_cell_rule.json",
    "uk_population_targets.json",
)


def test_no_uk_floor_of_ten_hides_outside_the_label() -> None:
    """The "<10" label is only one way to keep a floor of ten (the reviews of
    microcosm#1422). Over the UK runtime, the UK spec, the hand-written UK
    JSON, the UK tools and microcosm-data, this guard finds:

    - a comparison with ten, ``<``, ``<=``, ``>`` or ``>=`` on either side, and
      ``> 9`` or ``<= 9``, but not a format spec (``{x:<10}``, ``{x:>9}``,
      ``{x:0>9}``), ``10_000``, ``10.5``, ``10%`` or ``10 * x``;
    - a name with ``min``, ``minimum``, ``floor``, ``sdc``, ``small`` or
      ``threshold`` as one of its snake-case words, in any case, set to ten:
      a keyword, a YAML key, a single- or double-quoted key, or an assignment
      or default, annotated or not (``support_floor=10``, ``MIN_CELL = 10``,
      ``disclosure_floor: Final[int] = 10``), but not ``admin_count`` or
      ``minutes``;
    - prose: "fewer than", "under", "below", "less than" or "at least" before
      "ten" or "10", and "one to nine", also where a line break, a comment
      marker (``#`` or ``#:``) or a string's quotes split the words.

    A ten is ``10``, ``10.`` at a sentence's end or ``10.0``. Each it finds
    must be named in ``_KEPT_TENS``.
    """

    root = paths_for("microcosm-build").repository
    build = paths_for("microcosm-build").package / "src" / "microcosm" / "build"
    # A ten ends the number: not 10_000, 10% or 10.5, but "10." at a
    # sentence's end and 10.0 (the fourth review of microcosm#1422).
    ten = r"10\b(?![\d_%])(?!\.\d*[1-9])"
    nine = r"9\b(?![\d_%])(?!\.\d*[1-9])"
    # Not a format spec's width, with or without a fill: {x:<10}, {x:0>9}.
    spec = r"(?<![:{])(?<!:\S)"
    # Words a line break, a comment marker or a string's quotes split.
    gap = r"(?:[ \t]+|[ \t]*[\"']?[ \t]*\n[ \t]*(?:#:?|[rbfu]{0,2}[\"'])?[ \t]*)"
    # The floor words as snake-case words of a name, so support_floor and
    # MIN_CELL but not admin_count or minutes (the fifth review of
    # microcosm#1422).
    name = r"\b(?:\w*_)?(?:min|minimum|floors?|sdc|small|thresholds?)(?:_\w*)?\b"
    # An annotation without a comma outside its brackets: int, Final[int].
    annotation = r"(?:[\w.| \t]|\[[^\]\n]*\])+"
    floors = re.compile(
        rf"{spec}[<>]=?\s*{ten}(?!\s*\*)"
        r"|(?<![\w.])10(?:\.0+)?\s*[<>]=?(?![=\d])"
        rf"|{spec}(?:>|<=)\s*{nine}"
        rf"|{name}\s*[=:]\s*{ten}"
        rf"|{name}\s*:{annotation}=\s*{ten}"
        rf"|[\"']{name}[\"']\s*:\s*{ten}"
        rf"|\b(?:fewer{gap}than|under|below|less{gap}than|at{gap}least){gap}"
        rf"(?:ten\b|{ten})"
        rf"|\bone{gap}to{gap}nine\b",
        re.IGNORECASE | re.MULTILINE,
    )
    uk_tool = re.compile(r"(?:^|_)uk(?:_|$)")
    paths = (
        *(build / "uk_runtime").rglob("*.py"),
        *(build / "uk").glob("*.py"),
        *(build / "uk" / "spec").glob("*.yaml"),
        *(build / "uk" / name for name in _UK_HAND_WRITTEN_JSON),
        *(path for path in (root / "tools").rglob("*.py") if uk_tool.search(path.stem)),
        *(paths_for("microcosm-data").package / "src").rglob("*.py"),
    )
    found = []
    covered: dict[tuple[str, str], int] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(root).as_posix()
        kept = next(
            (
                entries
                for suffix, entries in _KEPT_TENS.items()
                if name.endswith(suffix)
            ),
            (),
        )
        for match in floors.finditer(text):
            phrase = next(
                (
                    phrase
                    for phrase, _ in kept
                    if any(
                        at.start() <= match.start() and at.end() >= match.end()
                        # Any whitespace between the words, so re-indenting
                        # the code keeps the phrase.
                        for at in re.finditer(
                            r"\s+".join(map(re.escape, phrase.split())), text
                        )
                    )
                ),
                None,
            )
            if phrase is None:
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                found.append(
                    f"{name}:{text.count(chr(10), 0, match.start()) + 1}: "
                    f"{' '.join(text[line_start:line_end].split())[:120]}"
                )
            else:
                key = (name, phrase)
                covered[key] = covered.get(key, 0) + 1
    assert found == []
    counts = {
        (suffix, phrase): count
        for suffix, entries in _KEPT_TENS.items()
        for phrase, count in entries
    }
    seen = {
        (suffix, phrase): sum(
            hits
            for (name, kept_phrase), hits in covered.items()
            if name.endswith(suffix) and kept_phrase == phrase
        )
        for suffix, phrase in counts
    }
    assert seen == counts


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
