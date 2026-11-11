from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.ods_tables import read_ods_tables


def test_reads_values_and_sheet_names(tmp_path, ods) -> None:
    path = ods.write(
        tmp_path / "book.ods",
        ods.sheet("Sheet A", [["heading", 1.5], [2.0, None]])
        + ods.sheet("Sheet B", [["only"]]),
    )

    tables = read_ods_tables(path, label="artifact")

    assert set(tables) == {"Sheet A", "Sheet B"}
    assert tables["Sheet A"].cell(0, 0) == "heading"
    assert tables["Sheet A"].cell(0, 1) == 1.5
    assert tables["Sheet A"].cell(1, 0) == 2.0
    assert tables["Sheet B"].column(0) == ("only",)


def test_expands_repeated_columns_and_rows(tmp_path, ods) -> None:
    """ODS collapses runs, so physical elements are not the logical grid."""
    tables_xml = (
        '<table:table table:name="R">'
        "<table:table-row>"
        f"{ods.cell(7.0, repeat=3)}{ods.cell('tail')}"
        "</table:table-row>"
        '<table:table-row table:number-rows-repeated="2">'
        f"{ods.cell(1.0)}"
        "</table:table-row>"
        "</table:table>"
    )
    path = ods.write(tmp_path / "repeat.ods", tables_xml)

    table = read_ods_tables(path, label="artifact")["R"]

    assert table.rows[0] == (7.0, 7.0, 7.0, "tail")
    assert table.rows[1] == (1.0,)
    assert table.rows[2] == (1.0,)


def test_preserves_suppression_markers(tmp_path, ods) -> None:
    """A withheld cell must arrive as its marker, not as zero or nothing."""
    path = ods.write(
        tmp_path / "suppressed.ods",
        ods.sheet("S", [["[Fewer than 1]", 420.0]]),
    )

    table = read_ods_tables(path, label="artifact")["S"]

    assert table.cell(0, 0) == "[Fewer than 1]"
    assert table.cell(0, 1) == 420.0


def test_reads_outside_the_grid_as_none(tmp_path, ods) -> None:
    path = ods.write(tmp_path / "small.ods", ods.sheet("S", [["a"]]))

    table = read_ods_tables(path, label="artifact")["S"]

    assert table.cell(5, 0) is None
    assert table.cell(0, 5) is None


def test_bounds_a_large_repeat_span(tmp_path, ods) -> None:
    """A trailing run padding the grid must not allocate without limit."""
    tables_xml = (
        '<table:table table:name="P">'
        f"<table:table-row>{ods.cell(1.0)}{ods.cell(None, repeat=1_000_000)}"
        "</table:table-row></table:table>"
    )
    path = ods.write(tmp_path / "pad.ods", tables_xml)

    table = read_ods_tables(path, label="artifact")["P"]

    assert table.rows[0] == (1.0,)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mimetype": "text/plain"}, "declares mimetype"),
        ({"store_mimetype": False}, "without compression"),
        ({"include_content": False}, "exactly one content.xml"),
    ],
)
def test_rejects_a_malformed_archive(tmp_path, ods, kwargs, message) -> None:
    path = ods.write(tmp_path / "bad.ods", ods.sheet("S", [["a"]]), **kwargs)

    with pytest.raises(ValueError, match=message):
        read_ods_tables(path, label="artifact")


def test_rejects_a_file_that_is_not_a_zip(tmp_path, ods) -> None:
    path = tmp_path / "plain.ods"
    path.write_bytes(b"not a zip")

    with pytest.raises(ValueError, match="not a readable zip archive"):
        read_ods_tables(path, label="artifact")


def test_names_the_artifact_in_errors(tmp_path, ods) -> None:
    """Errors say which source is malformed, not just that one is."""
    path = tmp_path / "plain.ods"
    path.write_bytes(b"not a zip")

    with pytest.raises(ValueError, match="HMRC capital gains table 3"):
        read_ods_tables(path, label="HMRC capital gains table 3")


def test_rejects_repeated_sheet_names(tmp_path, ods) -> None:
    path = ods.write(
        tmp_path / "dupe.ods",
        ods.sheet("Same", [["a"]]) + ods.sheet("Same", [["b"]]),
    )

    with pytest.raises(ValueError, match="repeats the sheet name"):
        read_ods_tables(path, label="artifact")
