"""Model totals beside official receipts, as a sourced diagnostic table.

A firm build's tax totals are compared with official receipts to see where
the data fall short. The comparison is a diagnostic, never a calibration
target, and it must not invent a gap:

- a total whose inputs the build leaves unset prints :data:`INPUT_NOT_POPULATED`
  rather than a -100% shortfall, so every total needs an explicit
  ``populated`` flag;
- a name with no official figure prints :data:`NO_OFFICIAL_FIGURE`;
- every official figure carries its publisher, title, edition, URL and access
  date, or the receipts file is refused.

The country supplies the receipts file, how money is formatted, and the
reasons each model measure differs from the official one.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "INPUT_NOT_POPULATED",
    "NO_MODEL_TOTAL",
    "NO_OFFICIAL_FIGURE",
    "RECEIPT_SOURCE_COLUMNS",
    "ComparisonRow",
    "Receipt",
    "build_comparison_rows",
    "compare_total",
    "read_receipts",
    "render_comparison_markdown",
]

INPUT_NOT_POPULATED: Final = "input not populated"
NO_OFFICIAL_FIGURE: Final = "no official figure fetched"
NO_MODEL_TOTAL: Final = "no model total supplied"

#: Columns every row with an official figure must fill.
RECEIPT_SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "measure",
    "source_publisher",
    "source_title",
    "edition",
    "url",
    "accessed",
)
#: Columns a receipts file carries besides the name and value columns.
_RECEIPT_TEXT_COLUMNS: Final[tuple[str, ...]] = (
    "period",
    *RECEIPT_SOURCE_COLUMNS,
    "note",
)


@dataclass(frozen=True)
class Receipt:
    """One official figure and where it comes from."""

    name: str
    official_value: float | None
    period: str
    measure: str
    source_publisher: str
    source_title: str
    edition: str
    url: str
    accessed: str
    note: str


@dataclass(frozen=True)
class ComparisonRow:
    """One row of the comparison table, already formatted."""

    name: str
    model: str
    official: str
    gap: str
    ratio: str
    reason: str


def read_receipts(
    path: Path, *, name_column: str, value_column: str
) -> dict[str, Receipt]:
    """Read a receipts CSV, refusing missing columns, repeats or unsourced figures.

    The file needs ``name_column``, ``value_column`` (blank when there is no
    official figure), ``period``, the :data:`RECEIPT_SOURCE_COLUMNS` and
    ``note``. Receipts keep the file's row order.
    """

    required = (name_column, value_column, *_RECEIPT_TEXT_COLUMNS)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        receipts: dict[str, Receipt] = {}
        for row in reader:
            name = row[name_column].strip()
            if name in receipts:
                raise ValueError(f"{path}: duplicate {name_column} {name!r}")
            raw = row[value_column].strip()
            value = float(raw) if raw else None
            if value is not None:
                unsourced = [c for c in RECEIPT_SOURCE_COLUMNS if not row[c].strip()]
                if unsourced:
                    raise ValueError(f"{path}: {name} has a figure but no {unsourced}")
            receipts[name] = Receipt(
                name=name,
                official_value=value,
                **{c: row[c].strip() for c in _RECEIPT_TEXT_COLUMNS},
            )
    return receipts


def compare_total(
    name: str,
    *,
    model_total: float | None,
    populated: bool,
    official_value: float | None,
    reasons: Sequence[str],
    format_value: Callable[[float], str],
) -> ComparisonRow:
    """One table row. No gap or ratio unless both sides are real figures.

    ``reasons`` are joined, in order, into the row's reason text.
    """

    official = (
        format_value(official_value)
        if official_value is not None
        else NO_OFFICIAL_FIGURE
    )
    if not populated:
        model = gap = ratio = INPUT_NOT_POPULATED
    elif model_total is None:
        model = gap = ratio = NO_MODEL_TOTAL
    else:
        model = format_value(model_total)
        if official_value is None:
            gap = ratio = NO_OFFICIAL_FIGURE
        else:
            difference = model_total - official_value
            gap = format_value(difference)
            if official_value:
                gap += f" ({difference / official_value:+.1%})"
                ratio = f"{model_total / official_value:.3f}"
            else:
                ratio = "official figure is zero"
    return ComparisonRow(name, model, official, gap, ratio, " ".join(reasons))


def build_comparison_rows(
    totals: Mapping[str, float | None],
    populated: Mapping[str, object],
    receipts: Mapping[str, Receipt],
    *,
    reasons_for: Callable[[str, Receipt | None], Sequence[str]],
    format_value: Callable[[float], str],
) -> list[ComparisonRow]:
    """Rows in receipts order, then any names only the model has, sorted.

    Every name in ``totals`` needs a true/false ``populated`` flag; without
    one an unset input would print as a -100% gap, so it is refused. A name
    in the receipts that the model left out entirely reads
    :data:`NO_MODEL_TOTAL`.
    """

    unflagged = sorted(
        name for name in totals if not isinstance(populated.get(name), bool)
    )
    if unflagged:
        raise ValueError(
            f"totals {unflagged} have no true/false populated flag; without it "
            "an unset input would print as a -100% gap"
        )
    names = list(receipts) + sorted(set(totals) - set(receipts))
    rows = []
    for name in names:
        total = totals.get(name)
        receipt = receipts.get(name)
        rows.append(
            compare_total(
                name,
                model_total=None if total is None else float(total),
                populated=bool(populated.get(name, True)),
                official_value=None if receipt is None else receipt.official_value,
                reasons=reasons_for(name, receipt),
                format_value=format_value,
            )
        )
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_comparison_markdown(
    rows: Sequence[ComparisonRow],
    receipts: Mapping[str, Receipt],
    *,
    title: str,
    name_header: str,
    note: str | None = None,
) -> str:
    """The comparison as markdown: a table, then a source line per receipt."""

    lines = [
        f"# {title}",
        "",
        "Gap is model less official (share of official); ratio is model / official.",
    ]
    if note:
        lines += ["", f"Model run note: {note}"]
    lines += [
        "",
        f"| {name_header} | Model total | Official | Gap | Ratio | Reason / caveat |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cells = (row.name, row.model, row.official, row.gap, row.ratio, row.reason)
        lines.append("| " + " | ".join(_cell(v) for v in cells) + " |")
    lines += ["", "## Sources", ""]
    for receipt in receipts.values():
        if receipt.official_value is None:
            lines.append(f"- `{receipt.name}`: {NO_OFFICIAL_FIGURE}. {receipt.note}")
        else:
            lines.append(
                f"- `{receipt.name}`: {receipt.source_publisher}, "
                f"{receipt.source_title}; {receipt.edition}; {receipt.url} "
                f"(accessed {receipt.accessed}). {receipt.note}"
            )
    return "\n".join(lines) + "\n"
