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
    "uk/spec/sources.yaml": (
        ("over at least 10 rows (a statistical floor", 1),
        # The zero-emission share's pooling floor, a statistical floor
        # (microcosm#1095 B3, J2).
        ("minimum_zero_emission_cars: 10", 1),
        ("a fifth on fewer than 10 zero-emission NTS cars", 1),
    ),
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


#: The columns no UK floor counts directly (microcosm#931): a household row
#: or a person's household is a row, not a survey household, once copies and
#: clones exist, and ``household_source_id`` is the support channel's
#: immediate parent. Floors count true sources through receipt_counts.
_ROW_IDENTITY_COLUMNS = frozenset(
    {"household_id", "person_household_id", "household_source_id"}
)
#: Counts of those columns that are no floor: structural checks that every
#: benefit unit sits in one household. Each entry names the phrase holding the
#: count and how many counts it covers in its file.
_KEPT_ROW_COUNTS: dict[str, tuple[tuple[str, int], ...]] = {
    "uk_runtime/uc_reporter_redraw.py": (
        (
            'counts = placements.groupby("person_benunit_id", sort=False)[ '
            '"person_household_id" ].nunique()',
            1,
        ),
    ),
    "tools/preflight_uk_frame_linkage.py": (
        (
            'households_per_benunit = person.groupby("person_benunit_id")[ '
            '"person_household_id" ].nunique()',
            1,
        ),
    ),
}
#: Calls that keep an identity's values, so a count of their result counts it.
_VALUE_KEEPING_METHODS = frozenset(
    {"to_numpy", "unique", "astype", "tolist", "to_list", "dropna"}
    | {"drop_duplicates", "copy", "sort_values"}
)
_VALUE_KEEPING_ATTRIBUTES = frozenset({"values", "loc", "iloc", "array"})
_VALUE_KEEPING_CALLS = frozenset(
    {"asarray", "array", "unique", "list", "set", "Series", "Index", "sorted"}
    | {"tuple"}
)


def _call_name(node) -> str | None:
    import ast

    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _row_identity(node, names: set[str]) -> bool:
    """Whether ``node`` holds a row identity column's values."""

    import ast

    while True:
        if isinstance(node, ast.Subscript):
            key = node.slice
            if isinstance(key, ast.Constant) and key.value in _ROW_IDENTITY_COLUMNS:
                return True
            node = node.value
        elif isinstance(node, ast.Attribute):
            if node.attr in _ROW_IDENTITY_COLUMNS:
                return True
            if node.attr not in _VALUE_KEEPING_ATTRIBUTES:
                return False
            node = node.value
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if isinstance(node.func, ast.Attribute) and name in _VALUE_KEEPING_METHODS:
                node = node.func.value
            elif name in _VALUE_KEEPING_CALLS and node.args:
                node = node.args[0]
            else:
                return False
        else:
            return isinstance(node, ast.Name) and node.id in names


def _counted_keys(node):
    """The keys a distinct count counts, or None for any other node."""

    import ast

    if not isinstance(node, ast.Call):
        return None
    name = _call_name(node)
    if name in {"distinct_count", "distinct_carriers"} and node.args:
        return node.args[0]
    if name in {"nunique", "value_counts"} and isinstance(node.func, ast.Attribute):
        return node.func.value
    if name == "len" and node.args and isinstance(node.args[0], ast.Call):
        inner = node.args[0]
        inner_name = _call_name(inner)
        if inner_name in {"set", "unique"} and inner.args:
            return inner.args[0]
        if inner_name in {"unique", "drop_duplicates"} and isinstance(
            inner.func, ast.Attribute
        ):
            return inner.func.value
    return None


def _row_identity_counts(tree) -> list[int]:
    """Lines that count a row identity column, scope by scope.

    A name holds an identity where it is assigned one in its own function or
    an enclosing one; a ``map`` through it holds what it maps to.
    """

    import ast

    found: list[int] = []

    def own_nodes(scope):
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                stack.extend(ast.iter_child_nodes(node))

    def scan(scope, inherited: set[str]) -> None:
        names = set(inherited)
        nodes = list(own_nodes(scope))
        assignments = [
            node
            for node in nodes
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        ]
        grew = True
        while grew:
            grew = False
            for node in assignments:
                if not _row_identity(node.value, names):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in names:
                        names.add(target.id)
                        grew = True
        for node in nodes:
            keys = _counted_keys(node)
            if keys is not None and _row_identity(keys, names):
                found.append(node.lineno)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scan(node, names)
            elif isinstance(node, ast.ClassDef):
                scan(node, set())

    scan(tree, set())
    return sorted(set(found))


def test_no_uk_floor_counts_rows_by_their_household_ids() -> None:
    """Every UK floor counts true source households (microcosm#931).

    Once support copies or geographic clones exist, ``household_id`` and
    ``person_household_id`` name rows, and ``household_source_id`` names the
    support channel's immediate parent, so a count of any of them reads a
    household's copies as sources of their own. Over the UK runtime and the UK
    tools, a distinct count (``distinct_count``, ``distinct_carriers``,
    ``nunique``, ``value_counts``, or ``len`` of a ``set`` or ``unique``) of
    such a column, or of a name holding one, must be named in
    ``_KEPT_ROW_COUNTS``; floors count through
    ``receipt_counts.record_source_households`` instead.
    """

    import ast

    root = paths_for("microcosm-build").repository
    build = paths_for("microcosm-build").package / "src" / "microcosm" / "build"
    uk_tool = re.compile(r"(?:^|_)uk(?:_|$)")
    paths = (
        *(build / "uk_runtime").rglob("*.py"),
        *(path for path in (root / "tools").rglob("*.py") if uk_tool.search(path.stem)),
    )
    found = []
    covered: dict[tuple[str, str], int] = {}
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8")
        name = path.relative_to(root).as_posix()
        kept = next(
            (
                entries
                for suffix, entries in _KEPT_ROW_COUNTS.items()
                if name.endswith(suffix)
            ),
            (),
        )
        lines = text.splitlines()
        for line in _row_identity_counts(ast.parse(text)):
            start = sum(len(row) + 1 for row in lines[: line - 1])
            end = start + len(lines[line - 1])
            phrase = next(
                (
                    phrase
                    for phrase, _ in kept
                    if any(
                        at.start() <= end and at.end() >= start
                        for at in re.finditer(
                            r"\s*".join(map(re.escape, phrase.split())), text
                        )
                    )
                ),
                None,
            )
            if phrase is None:
                found.append(f"{name}:{line}: {lines[line - 1].strip()[:120]}")
            else:
                covered[(name, phrase)] = covered.get((name, phrase), 0) + 1
    assert found == []
    expected = {
        (suffix, phrase): count
        for suffix, entries in _KEPT_ROW_COUNTS.items()
        for phrase, count in entries
    }
    seen = {
        (suffix, phrase): sum(
            hits
            for (name, kept_phrase), hits in covered.items()
            if name.endswith(suffix) and kept_phrase == phrase
        )
        for suffix, phrase in expected
    }
    assert seen == expected
