"""The England-and-Wales Band D tool: pins, joins, quantisation, refusals (microcosm#931).

Every test but the last reads no real registration; those that read inputs
write a minimal ODS standing for MHCLG's Table 8 and a CSV standing for
StatsWales' export, over the real local authority roster with invented
amounts. The published layout (sheet, columns, row selectors, the declared
exclusion) is restated here rather than read from the tool, so a slip in the
tool's declaration fails here too. The last test rebuilds the committed
resource with ``--check`` when the real registrations are mounted, and is
skipped otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from microcosm.build.uk_runtime import ods_tables
from microcosm.build.uk_runtime.council_tax_band_d import (
    UK_BAND_D_ENGLAND_WALES_RESOURCE,
)
from tools import build_uk_council_tax_band_d as tool

#: The real inputs, when mounted: each registration's file and its
#: ``manifest.yaml`` at the pinned Chronicle commit.
MOUNTED_INPUTS = {
    "--england-ods": "CHRONICLE_UK_BAND_D_ENGLAND_ODS",
    "--england-manifest": "CHRONICLE_UK_BAND_D_ENGLAND_MANIFEST",
    "--wales-csv": "CHRONICLE_UK_BAND_D_WALES_CSV",
    "--wales-manifest": "CHRONICLE_UK_BAND_D_WALES_MANIFEST",
}

ENGLAND_SHEET = "Table_8"
ENGLAND_HEADER = [
    "E Code",
    "ONS Code",
    "Authority",
    "Region",
    "Class",
    "Area",
    *(f"Band {band}" for band in "ABCDEFGH"),
    "Notes",
]
CODE, BAND_D, NOTES = (
    ENGLAND_HEADER.index(name) for name in ("ONS Code", "Band D", "Notes")
)
WALES_COLUMNS = ["Data values", "Data description", "Row", "Year", "Authority", "Notes"]
WALES_TOTAL = "Total CT for billing authority area"
WALES_UNIT = "\N{POUND SIGN} per band D"
WALES_EXCLUSION = "Total Unitary Authorities"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------------
# Synthetic inputs


def _england_values() -> tuple[dict[str, float], dict[str, str]]:
    """Invented amounts by English roster code, as stored and as pence."""

    english, _ = tool.roster()
    stored: dict[str, float] = {}
    pence: dict[str, str] = {}
    for index, code in enumerate(sorted(english)):
        # A quarter is an exact binary float; seven hundredths is not.
        cents = 25 if index % 4 == 0 else 7
        stored[code] = (1500 + index) + cents / 100
        pence[code] = f"{1500 + index}.{cents:02d}"
    # A float a hair off its pence, as a binary sheet can store one.
    first = sorted(english)[1]
    stored[first] = 1234.560000000001
    pence[first] = "1234.56"
    return stored, pence


def _england_rows(stored: dict[str, object]) -> list[list[object]]:
    rows: list[list[object]] = [
        ["Table 8: Area council tax for a dwelling occupied by two adults (invented)"],
        ["The data in this table are reported in pounds."],
        list(ENGLAND_HEADER),
    ]
    for index, (code, value) in enumerate(stored.items()):
        row: list[object] = [None] * (BAND_D + 1)
        row[0], row[CODE], row[2], row[BAND_D] = f"E{index:04d}", code, "Area", value
        rows.append(row)
    return rows


def _row_xml(ods, row: list[object]) -> str:
    return f"<table:table-row>{''.join(ods.cell(value) for value in row)}</table:table-row>"


def _write_england(tmp_path: Path, ods, rows: list[list[object]]) -> Path:
    return ods.write(
        tmp_path / "Table_8_24-25.ods",
        ods.sheet(ENGLAND_SHEET, rows) + ods.sheet("Notes", [["Notes"]]),
    )


def _wales_rows() -> tuple[list[dict[str, str]], dict[str, str]]:
    """Invented rows: the selected ones, distractors, and the declared exclusion."""

    _, welsh = tool.roster()
    rows: list[dict[str, str]] = []
    expected: dict[str, str] = {}
    for index, name in enumerate(sorted(welsh)):
        text = "2000" if index == 0 else f"{1800 + index}.{123456 + index}"
        expected[welsh[name]] = text
        for year, row, unit, value in (
            ("2024-25", WALES_TOTAL, WALES_UNIT, text),
            ("2023-24", WALES_TOTAL, WALES_UNIT, "1.5"),
            ("2024-25", "Police council tax within county area", WALES_UNIT, "1.5"),
            ("2024-25", WALES_TOTAL, "Percentage change", "1.5"),
        ):
            rows.append(
                {
                    "Data values": value,
                    "Data description": unit,
                    "Row": row,
                    "Year": year,
                    "Authority": name,
                    "Notes": "",
                }
            )
    rows.append(
        {
            "Data values": "1999.5",
            "Data description": WALES_UNIT,
            "Row": WALES_TOTAL,
            "Year": "2024-25",
            "Authority": WALES_EXCLUSION,
            "Notes": "",
        }
    )
    return rows, expected


def _write_wales(
    tmp_path: Path, rows: list[dict[str, str]], *, columns: list[str] | None = None
) -> Path:
    path = tmp_path / "statswales_band_d.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns or WALES_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _selected(row: dict[str, str], name: str | None = None) -> bool:
    return (
        row["Year"] == "2024-25"
        and row["Row"] == WALES_TOTAL
        and row["Data description"] == WALES_UNIT
        and (name is None or row["Authority"] == name)
    )


def _manifest(pin: dict) -> dict:
    """A manifest laid out as Chronicle's ``manifest.yaml`` is, for a pin."""

    return {
        "source_id": pin["chronicle_source_id"],
        "package_id": pin["chronicle_package_id"],
        "table": pin["table"],
        "files": {
            pin["chronicle_year"]: {
                "filename": pin["filename"],
                "source_url": pin["url"],
                "sha256": pin["sha256"],
                "size_bytes": pin["size_bytes"],
            }
        },
        "registration_kind": pin["registration_kind"],
    }


def _write_manifest(path: Path, manifest: dict) -> Path:
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Joins and quantisation


def test_reads_joins_and_quantises_over_the_roster(tmp_path, ods) -> None:
    stored, pence = _england_values()
    england = _write_england(tmp_path, ods, _england_rows(stored))
    rows, welsh = _wales_rows()
    band_d, receipts = tool.band_d_by_code(england, _write_wales(tmp_path, rows))

    english_codes, welsh_codes = tool.roster()
    assert set(band_d) == set(english_codes) | set(welsh_codes.values())
    assert list(band_d) == sorted(band_d)
    assert band_d == {**pence, **welsh}
    exact = sum(1 for index in range(len(english_codes)) if index % 4 == 0)
    assert receipts == {
        "authorities": {"england": len(english_codes), "wales": len(welsh_codes)},
        "england_values_quantised_to_pence": {
            "exact_binary": exact,
            "within_tolerance": len(english_codes) - exact,
            "tolerance": "1e-9",
        },
        "wales_excluded_names": [WALES_EXCLUSION],
    }


@pytest.mark.parametrize(
    ("stored", "kept"),
    [
        (1234.560000000001, "1234.56"),
        (1234.5600000005, "1234.56"),
        (1457.0, "1457.00"),
        (4321.5, "4321.50"),
        (0.05, "0.05"),
    ],
)
def test_keeps_the_pence_decimal_within_the_tolerance(
    tmp_path, ods, stored, kept
) -> None:
    values, pence = _england_values()
    code = next(iter(values))
    values[code] = stored
    path = _write_england(tmp_path, ods, _england_rows(values))
    english, _ = tool.roster()

    band_d, _ = tool.england_band_d(path, english_codes=english)

    assert band_d == {**pence, code: kept}


def test_an_ods_repeating_a_million_empty_rows_reads_within_the_shared_bound(
    tmp_path, ods
) -> None:
    """Published sheets pad their grid to its 1,048,576-row limit.

    They do it with long runs of empty rows, which the shared reader bounds.
    """

    stored, pence = _england_values()
    rows = _england_rows(stored)
    run = (
        '<table:table-row table:number-rows-repeated="1048576">'
        '<table:table-cell table:number-columns-repeated="16384"/></table:table-row>'
    )
    body = (
        "".join(_row_xml(ods, row) for row in rows[:3])
        + run
        + "".join(_row_xml(ods, row) for row in rows[3:])
        + run
    )
    path = ods.write(
        tmp_path / "padded.ods",
        f'<table:table table:name="{ENGLAND_SHEET}">{body}</table:table>',
    )
    english, _ = tool.roster()

    table = ods_tables.read_ods_tables(path, label="padded")[ENGLAND_SHEET]
    band_d, _ = tool.england_band_d(path, english_codes=english)

    # The run between header and data expands to the shared bound; the
    # trailing one pads nothing.
    assert len(table.rows) == len(rows) + ods_tables._MAX_REPEAT
    assert band_d == pence


# ----------------------------------------------------------------------------
# Refusals


def _replace(position: int, value: object) -> Callable[[list[list[object]]], None]:
    def mutate(rows: list[list[object]]) -> None:
        rows[3][position] = value

    return mutate


def _note(rows: list[list[object]]) -> None:
    rows[3].extend([None] * (NOTES + 1 - len(rows[3])))
    rows[3][NOTES] = "[note c]"


def _without_header(rows: list[list[object]]) -> None:
    rows[2][CODE] = "Code"


def _without_code(rows: list[list[object]]) -> None:
    rows.append([None, None, None, None, None, None, None, None, None, 1500.0])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "lacks the roster code"),
        (lambda rows: rows.append(list(rows[3])), "repeats"),
        (_replace(CODE, "N09000001"), "Northern Ireland code 'N09000001'"),
        (_replace(CODE, "E07000150"), "not an English code on the local authority"),
        (_replace(CODE, "W06000015"), "not an English code on the local authority"),
        (_replace(BAND_D, 1500.0700000025), "within 1e-9 of a whole number of pence"),
        (_replace(BAND_D, 1500.075), "within 1e-9 of a whole number of pence"),
        (_replace(BAND_D, "[x]"), "is not a number"),
        (_replace(BAND_D, 0.0), "is not a positive finite amount"),
        (_replace(BAND_D, -1500.0), "is not a positive finite amount"),
        (_note, "carries a note"),
        (_without_header, "exactly one header row"),
        (_without_code, "has no 'ONS Code'"),
    ],
    ids=[
        "missing",
        "duplicate",
        "northern-ireland",
        "off-roster",
        "welsh-code",
        "off-pence",
        "half-penny",
        "text",
        "zero",
        "negative",
        "note",
        "no-header",
        "no-code",
    ],
)
def test_refuses_a_table_8_outside_the_closed_world(
    tmp_path, ods, mutate, message
) -> None:
    stored, _ = _england_values()
    rows = _england_rows(stored)
    mutate(rows)
    path = _write_england(tmp_path, ods, rows)
    english, _ = tool.roster()

    with pytest.raises(ValueError, match=message) as refusal:
        tool.england_band_d(path, english_codes=english)
    # A refusal names columns, codes or names, never an amount.
    text = str(refusal.value)
    assert "1500" not in text and "1234" not in text, text
    assert not re.search(r"\d\.\d", text), text


def _rename(old: str, new: str) -> Callable[[list[dict[str, str]]], None]:
    def mutate(rows: list[dict[str, str]]) -> None:
        for row in rows:
            if row["Authority"] == old:
                row["Authority"] = new

    return mutate


def _set_selected(field: str, value: str) -> Callable[[list[dict[str, str]]], None]:
    def mutate(rows: list[dict[str, str]]) -> None:
        next(row for row in rows if _selected(row))[field] = value

    return mutate


def _duplicate(rows: list[dict[str, str]]) -> None:
    rows.append(dict(next(row for row in rows if _selected(row))))


def _drop(name: str) -> Callable[[list[dict[str, str]]], None]:
    def mutate(rows: list[dict[str, str]]) -> None:
        rows[:] = [row for row in rows if not _selected(row, name)]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_rename("Isle of Anglesey", "Anglesey"), "'Anglesey' matches no Welsh"),
        (_rename("Cardiff", "Belfast"), "'Belfast' matches no Welsh"),
        (_duplicate, "repeats"),
        (_drop("Powys"), r"lacks the roster authority or authorities \['Powys'\]"),
        (_drop(WALES_EXCLUSION), "declared exclusion"),
        (_set_selected("Notes", "Provisional"), "carries a note"),
        (_set_selected("Data values", "4,321.07"), "not a positive decimal amount"),
        (_set_selected("Data values", "-5.00"), "not a positive decimal amount"),
        (_set_selected("Data values", "0.000000"), "not a positive decimal amount"),
        (_set_selected("Data values", ""), "not a positive decimal amount"),
    ],
    ids=[
        "unmatched-name",
        "northern-ireland-name",
        "duplicate",
        "missing",
        "no-exclusion",
        "note",
        "thousands-separator",
        "negative",
        "zero",
        "blank",
    ],
)
def test_refuses_welsh_rows_outside_the_closed_world(tmp_path, mutate, message) -> None:
    rows, _ = _wales_rows()
    mutate(rows)
    path = _write_wales(tmp_path, rows)
    _, welsh = tool.roster()

    with pytest.raises(ValueError, match=message) as refusal:
        tool.wales_band_d(path, welsh_codes_by_name=welsh)
    # A refusal names columns or names, never an amount.
    text = str(refusal.value)
    assert "4321" not in text and "1801" not in text, text
    assert not re.search(r"\d", text), text


def test_refuses_a_csv_with_other_columns(tmp_path) -> None:
    rows, _ = _wales_rows()
    columns = [*WALES_COLUMNS[:-1], "Footnote"]
    for row in rows:
        row["Footnote"] = row.pop("Notes")
    path = _write_wales(tmp_path, rows, columns=columns)
    _, welsh = tool.roster()

    with pytest.raises(ValueError, match="must have exactly the columns"):
        tool.wales_band_d(path, welsh_codes_by_name=welsh)


# ----------------------------------------------------------------------------
# Pins


def test_refuses_a_file_whose_bytes_are_not_the_pinned_registration(
    tmp_path, ods
) -> None:
    stored, _ = _england_values()
    path = _write_england(tmp_path, ods, _england_rows(stored))
    pin = tool.SOURCES["england"]
    manifest = _write_manifest(tmp_path / "manifest.yaml", _manifest(pin))

    with pytest.raises(ValueError, match="differs from its pinned sha256 and size"):
        tool.verify_source("england", path, manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("package_id", "mhclg-council-tax-levels-england-2025-26", "package_id"),
        ("table", "Table 9", "table"),
        ("registration_kind", "facts", "registration_kind"),
        ("sha256", "0" * 64, r"files\.2024\.sha256"),
        ("size_bytes", 1, r"files\.2024\.size_bytes"),
        ("source_url", "https://example.org/Table_8.ods", r"files\.2024\.source_url"),
    ],
)
def test_refuses_a_manifest_that_is_not_the_pinned_registration(
    tmp_path, ods, field, value, message
) -> None:
    stored, _ = _england_values()
    path = _write_england(tmp_path, ods, _england_rows(stored))
    pin = {
        **tool.SOURCES["england"],
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }
    manifest = _manifest(pin)
    if field in manifest:
        manifest[field] = value
    else:
        manifest["files"][pin["chronicle_year"]][field] = value
    manifest_path = _write_manifest(tmp_path / "manifest.yaml", manifest)

    tool.verify_source(
        "england",
        path,
        _write_manifest(tmp_path / "pinned.yaml", _manifest(pin)),
        pin=pin,
    )
    with pytest.raises(ValueError, match=f"manifest's {message} differs from the pin"):
        tool.verify_source("england", path, manifest_path, pin=pin)


def test_the_pins_are_json_native_and_name_one_chronicle_commit() -> None:
    assert json.loads(json.dumps(tool.SOURCES)) == tool.SOURCES
    assert set(tool.SOURCES) == {"england", "wales"}
    for pin in tool.SOURCES.values():
        assert pin["chronicle_manifest"]["commit"] == tool.CHRONICLE_COMMIT
        assert pin["registered_by"]["pull_request"] == "PolicyEngine/chronicle#413"
        assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])
    assert tool.SOURCES["wales"]["selector"]["columns"] == WALES_COLUMNS
    assert tool.SOURCES["wales"]["selector"]["rows"] == {
        "Year": "2024-25",
        "Row": WALES_TOTAL,
        "Data description": WALES_UNIT,
    }
    assert tool.SOURCES["wales"]["selector"]["excluded_names"] == [WALES_EXCLUSION]


# ----------------------------------------------------------------------------
# The command


def _pinned_inputs(
    tmp_path: Path, ods, monkeypatch
) -> tuple[list[str], dict[str, str]]:
    """Synthetic inputs, their manifests, and the tool pinned to them."""

    stored, pence = _england_values()
    rows, welsh = _wales_rows()
    files = {
        "england": _write_england(tmp_path, ods, _england_rows(stored)),
        "wales": _write_wales(tmp_path, rows),
    }
    argv: list[str] = []
    for name, path in files.items():
        pin = {
            **tool.SOURCES[name],
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        monkeypatch.setitem(tool.SOURCES, name, pin)
        manifest = _write_manifest(tmp_path / f"{name}.manifest.yaml", _manifest(pin))
        kind = "ods" if name == "england" else "csv"
        argv += [f"--{name}-{kind}", str(path), f"--{name}-manifest", str(manifest)]
    return argv, {**pence, **welsh}


def test_writes_sorted_ascii_json_and_checks_all_but_the_build_date(
    tmp_path, ods, monkeypatch
) -> None:
    argv, expected = _pinned_inputs(tmp_path, ods, monkeypatch)
    out = tmp_path / UK_BAND_D_ENGLAND_WALES_RESOURCE

    assert tool.main([*argv, "--out", str(out), "--built-on", "2026-11-13"]) == 0

    text = out.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert (
        text == json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True) + "\n"
    )
    assert text.isascii()
    assert {
        key: payload[key]
        for key in (
            "schema_version",
            "kind",
            "resource",
            "method",
            "built_on",
            "fiscal_year",
            "local_authority_vintage",
        )
    } == {
        "schema_version": 1,
        "kind": "uk_council_tax_band_d_england_wales",
        "resource": "council_tax_band_d_england_wales_2024_25.json",
        "method": "uk-council-tax-band-d-v1",
        "built_on": "2026-11-13",
        "fiscal_year": "2024-25",
        "local_authority_vintage": "2023_april_lad",
    }
    assert payload["band_d"] == expected
    assert payload["sources"] == tool.SOURCES
    assert payload["licence"]["licence"] == "Open Government Licence v3.0"

    check = [*argv, "--out", str(out), "--check"]
    assert tool.main([*check, "--built-on", "2027-01-01"]) == 0
    code = next(iter(expected))
    out.write_text(
        tool.render({**payload, "band_d": {**payload["band_d"], code: "1.00"}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="is stale"):
        tool.main([*check, "--built-on", "2026-11-13"])
    out.write_text(text.replace("\n ", "\n  "), encoding="utf-8")
    with pytest.raises(SystemExit, match="is stale"):
        tool.main([*check, "--built-on", "2026-11-13"])
    out.unlink()
    with pytest.raises(SystemExit, match="is stale"):
        tool.main([*check, "--built-on", "2026-11-13"])


@pytest.mark.parametrize("built_on", ["13/11/2026", "2026-11-1", "2026-13-01", ""])
def test_the_build_date_is_an_explicit_iso_date(
    tmp_path, ods, monkeypatch, built_on
) -> None:
    argv, _ = _pinned_inputs(tmp_path, ods, monkeypatch)

    with pytest.raises(SystemExit):
        tool.main([*argv, "--out", str(tmp_path / "out.json"), "--built-on", built_on])
    assert not (tmp_path / "out.json").exists()


def test_the_command_refuses_inputs_that_are_not_the_registrations(
    tmp_path, ods
) -> None:
    stored, _ = _england_values()
    rows, _ = _wales_rows()
    argv = [
        "--england-ods",
        str(_write_england(tmp_path, ods, _england_rows(stored))),
        "--england-manifest",
        str(_write_manifest(tmp_path / "e.yaml", _manifest(tool.SOURCES["england"]))),
        "--wales-csv",
        str(_write_wales(tmp_path, rows)),
        "--wales-manifest",
        str(_write_manifest(tmp_path / "w.yaml", _manifest(tool.SOURCES["wales"]))),
    ]

    with pytest.raises(SystemExit, match="Refused: The england source differs"):
        tool.main(
            [*argv, "--out", str(tmp_path / "out.json"), "--built-on", "2026-11-13"]
        )
    assert not (tmp_path / "out.json").exists()


def test_check_passes_on_the_committed_resource_when_the_registrations_are_mounted() -> (
    None
):
    paths = {
        flag: os.environ.get(variable) for flag, variable in MOUNTED_INPUTS.items()
    }
    if not all(value and Path(value).is_file() for value in paths.values()):
        pytest.skip(
            "the Band D registrations are not mounted (set "
            f"{', '.join(MOUNTED_INPUTS.values())})"
        )
    argv = [item for flag, value in paths.items() for item in (flag, str(value))]

    assert tool.main([*argv, "--built-on", "2026-11-13", "--check"]) == 0
