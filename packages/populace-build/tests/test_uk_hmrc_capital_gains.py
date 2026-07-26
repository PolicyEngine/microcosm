from __future__ import annotations

from pathlib import Path

import pytest

from populace.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRC_CGT_JOINT_ODS_FILENAME,
    HMRC_CGT_JOINT_ODS_SHA256,
    HMRC_CGT_JOINT_ODS_SIZE_BYTES,
    HMRC_CGT_JOINT_SHEET_NAMES,
    HMRC_CGT_TOTAL_GAINS_GBP,
    HMRC_CGT_TOTAL_INDIVIDUALS,
    materialize_hmrc_capital_gains_joint_distribution,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PINNED_ODS_PATH = _REPO_ROOT / "inputs" / "hmrc" / HMRC_CGT_JOINT_ODS_FILENAME

_NOTE_ROWS = 9
_HEADER = ["Range of gain (Lower limit £)"] + [
    heading
    for bound in (*HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, "All")
    for heading in (f"Number of individuals ({bound})", f"Amounts of gains ({bound})")
]


def _synthetic_rows(*, notes: int = _NOTE_ROWS, suppress: bool = False):
    """A sheet shaped like the published one, with small round numbers."""
    rows: list[list[object]] = [[f"note {index}"] for index in range(notes)]
    rows.append(list(_HEADER))
    for band_index, lower_bound in enumerate(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS):
        row: list[object] = [float(lower_bound)]
        for income_index in range(len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)):
            count = float(band_index + income_index + 1)
            amount = float(10 * (band_index + 1) + income_index)
            if suppress and band_index == 0 and income_index == 0:
                row.extend(["[Fewer than 1]", "[x]"])
            else:
                row.extend([count, amount])
        row.extend([0.0, 0.0])
        rows.append(row)

    published_amount = sum(
        float(10 * (band_index + 1) + income_index)
        for band_index in range(len(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS))
        for income_index in range(len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS))
        if not (suppress and band_index == 0 and income_index == 0)
    )
    rows.append(["All"] + [0.0] * (2 * len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)))
    rows[-1].extend([1.0, published_amount])
    return rows


def _synthetic_ods(
    ods,
    tmp_path,
    *,
    notes: int = _NOTE_ROWS,
    suppress: bool = False,
    name: str = "cgt.ods",
) -> Path:
    sheet = HMRC_CGT_JOINT_SHEET_NAMES["2023-24"]
    return ods.write(
        tmp_path / name,
        ods.sheet(sheet, _synthetic_rows(notes=notes, suppress=suppress)),
    )


def _load(path: Path, **kwargs):
    return materialize_hmrc_capital_gains_joint_distribution(
        path, verify_fingerprint=False, **kwargs
    )


def test_parses_every_published_cell(tmp_path, ods) -> None:
    distribution = _load(_synthetic_ods(ods, tmp_path))

    assert len(distribution.cells) == len(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS) * len(
        HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )
    assert {cell.gain_lower_bound for cell in distribution.cells} == set(
        HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    )
    assert {cell.income_lower_bound for cell in distribution.cells} == set(
        HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    )


def test_converts_source_units_to_people_and_pounds(tmp_path, ods) -> None:
    """Counts publish in thousands and amounts in £ millions."""
    distribution = _load(_synthetic_ods(ods, tmp_path))

    cell = distribution.cell(gain_lower_bound=0, income_lower_bound=0)

    assert cell.individuals == 1_000.0
    assert cell.gains == 10_000_000.0


def test_reports_suppressed_cells_as_unknown_rather_than_zero(tmp_path, ods) -> None:
    """A withheld count means "fewer than 500 people", not "nobody"."""
    distribution = _load(_synthetic_ods(ods, tmp_path, suppress=True))

    cell = distribution.cell(gain_lower_bound=0, income_lower_bound=0)

    assert cell.individuals is None
    assert cell.individuals_suppressed
    assert cell.gains is None
    assert cell.gains_suppressed
    assert distribution.gains_by_band()[0] > 0


def test_finds_the_header_whatever_the_note_count(tmp_path, ods) -> None:
    """2020-21 carries one fewer note than 2023-24."""
    fewer_notes = _load(
        _synthetic_ods(ods, tmp_path, notes=_NOTE_ROWS - 1, name="fewer.ods")
    )
    more_notes = _load(
        _synthetic_ods(ods, tmp_path, notes=_NOTE_ROWS + 2, name="more.ods")
    )

    assert len(fewer_notes.cells) == len(more_notes.cells)
    assert fewer_notes.cell(
        gain_lower_bound=5_000_000, income_lower_bound=200_000
    ) == more_notes.cell(gain_lower_bound=5_000_000, income_lower_bound=200_000)


def test_rejects_an_unpublished_tax_year(tmp_path, ods) -> None:
    with pytest.raises(ValueError, match="publishes"):
        _load(_synthetic_ods(ods, tmp_path), tax_year="2019-20")


def test_rejects_a_missing_sheet(tmp_path, ods) -> None:
    path = ods.write(tmp_path / "other.ods", ods.sheet("Contents", [["a"]]))

    with pytest.raises(ValueError, match="has no sheet"):
        _load(path)


def test_rejects_a_moved_header(tmp_path, ods) -> None:
    rows = _synthetic_rows()
    rows[_NOTE_ROWS][0] = "Something else entirely"
    path = ods.write(
        tmp_path / "moved.ods",
        ods.sheet(HMRC_CGT_JOINT_SHEET_NAMES["2023-24"], rows),
    )

    with pytest.raises(ValueError, match="no column headed"):
        _load(path)


def test_rejects_a_reordered_band(tmp_path, ods) -> None:
    rows = _synthetic_rows()
    rows[_NOTE_ROWS + 1][0] = 999_999.0
    path = ods.write(
        tmp_path / "reordered.ods",
        ods.sheet(HMRC_CGT_JOINT_SHEET_NAMES["2023-24"], rows),
    )

    with pytest.raises(ValueError, match="opens band"):
        _load(path)


def test_rejects_an_artifact_that_is_not_the_pinned_one(tmp_path, ods) -> None:
    """Identity is checked before parsing, so a wrong file fails as itself."""
    path = _synthetic_ods(ods, tmp_path)

    with pytest.raises(ValueError, match="bytes, not the pinned"):
        materialize_hmrc_capital_gains_joint_distribution(path)


@pytest.mark.skipif(
    not _PINNED_ODS_PATH.is_file(),
    reason="reviewed HMRC capital gains ODS is an optional local input",
)
class TestPinnedPublication:
    def test_matches_the_pinned_fingerprint(self) -> None:
        distribution = materialize_hmrc_capital_gains_joint_distribution(
            _PINNED_ODS_PATH
        )

        assert distribution.source.sha256 == HMRC_CGT_JOINT_ODS_SHA256
        assert distribution.source.size_bytes == HMRC_CGT_JOINT_ODS_SIZE_BYTES
        assert distribution.source.local_path == _PINNED_ODS_PATH.resolve()

    def test_reproduces_the_published_2023_24_totals(self) -> None:
        distribution = materialize_hmrc_capital_gains_joint_distribution(
            _PINNED_ODS_PATH
        )

        assert distribution.total_individuals == HMRC_CGT_TOTAL_INDIVIDUALS
        assert distribution.total_gains == HMRC_CGT_TOTAL_GAINS_GBP

    @pytest.mark.parametrize("tax_year", sorted(HMRC_CGT_JOINT_SHEET_NAMES))
    def test_every_published_year_parses(self, tax_year: str) -> None:
        distribution = materialize_hmrc_capital_gains_joint_distribution(
            _PINNED_ODS_PATH, tax_year=tax_year
        )

        assert len(distribution.cells) == 60
        assert distribution.total_gains > 0
        # Suppressed cells hold a rounding-scale residual, not real mass.
        assert abs(distribution.unpublished_gains) < 50_000_000

    def test_carries_the_top_tail_the_percentile_source_cannot_reach(self) -> None:
        """The reason for reading this table rather than a percentile one."""
        distribution = materialize_hmrc_capital_gains_joint_distribution(
            _PINNED_ODS_PATH
        )
        by_band = distribution.gains_by_band()

        assert by_band[5_000_000] > 20_000_000_000
        top_two = by_band[2_000_000] + by_band[5_000_000]
        assert top_two / distribution.total_gains > 0.5
