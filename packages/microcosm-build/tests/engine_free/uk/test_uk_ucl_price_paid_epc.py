"""UCL's Price Paid–EPC linked file and certificates, streamed (microcosm#1524)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import high_value_home_stock as hv
from microcosm.build.uk_runtime import stock_cell_declarations as declared
from microcosm.build.uk_runtime import ucl_price_paid_epc as ucl

_COLUMNS = ucl.LinkedColumns(
    transaction_id="transactionid",
    date="dateoftransfer",
    floor_area="tfarea",
    age_band="CONSTRUCTION_AGE_BAND",
    postcode="postcode",
    paon="paon",
    saon="saon",
)
_BOUNDS = declared.FLOOR_AREA_BOUNDS_M2
_BANDS = declared.EPC_AGE_BANDS

_LINKED_HEADER = [
    "transactionid",
    "price",
    "dateoftransfer",
    "postcode",
    "paon",
    "saon",
    "tfarea",
    "CONSTRUCTION_AGE_BAND",
]
_LINKED_ROWS = [
    # One dwelling sold twice: the floor area is the later sale's, the age
    # band the earlier one's, since the later certificate has none.
    [
        "{AAAA-0001}",
        "100000",
        "2010-03-05 00:00",
        "AB1 2CD",
        "1",
        "",
        "80",
        "England and Wales: 1930-1949",
    ],
    ["{AAAA-0002}", "150000", "2019-06-01 00:00", "AB12CD", "1", "", "85", "NO DATA!"],
    # A floor area below the bounds is a data error.
    [
        "{AAAA-0003}",
        "90000",
        "2015-01-02 00:00",
        "AB1 2CD",
        "Flat 2",
        "Rose Court",
        "5",
        "2012 onwards",
    ],
    # A British date and an open band.
    [
        "{AAAA-0004}",
        "300000",
        "05/03/2016",
        "EF3 4GH",
        "The Grange",
        "",
        "120.5",
        "before 1900",
    ],
]


def _linked_csv() -> bytes:
    lines = [",".join(_LINKED_HEADER)] + [
        ",".join(f'"{v}"' for v in row) for row in _LINKED_ROWS
    ]
    return ("﻿" + "\n".join(lines) + "\n").encode("utf-8")


def _bundle(
    path: Path, inner_name: str, members: dict[str, bytes], *, stored: bool = True
) -> Path:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr(
            zipfile.ZipInfo(inner_name),
            inner.getvalue(),
            compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
        )
        outer.writestr("856542_readme.txt", "Fields: transactionid, tfarea ...\n")
    return path


def _key(postcode: str, paon: str, saon: str) -> int:
    return int(
        hv.dwelling_key(pd.Series([postcode]), pd.Series([paon]), pd.Series([saon]))[0]
    )


@pytest.mark.parametrize("stored", [True, False])
def test_linked_attributes_take_each_attribute_from_the_latest_usable_sale(
    tmp_path: Path, stored: bool
) -> None:
    bundle = _bundle(
        tmp_path / "bundle_1.zip",
        ucl.LINKED_MEMBER,
        {"tranall_link_02122023.csv": _linked_csv()},
        stored=stored,
    )
    out, receipts = ucl.linked_attributes(
        bundle,
        _COLUMNS,
        age_bands=_BANDS,
        floor_area_bounds=_BOUNDS,
        key_function=hv.dwelling_key,
        block_size=1 << 10,
    )
    rows = out.set_index("key")
    house = rows.loc[_key("AB12CD", "1", "")]
    assert house["floor_area"] == 85 and house["floor_area_yyyymm"] == 201906
    assert house["age_band"] == 2 and house["age_band_yyyymm"] == 201003
    flat = rows.loc[_key("AB1 2CD", "FLAT 2", "ROSE COURT")]
    assert np.isnan(flat["floor_area"]) and flat["age_band"] == len(_BANDS) - 1
    grange = rows.loc[_key("EF34GH", "THE GRANGE", "")]
    assert grange["floor_area"] == pytest.approx(120.5)
    assert grange["age_band"] == 0 and grange["age_band_yyyymm"] == 201603
    assert list(out["key"]) == sorted(out["key"])
    assert receipts["rows"] == 4 and receipts["dwellings"] == 3
    assert receipts["dwellings_with_floor_area"] == 2
    assert receipts["join"] == "address key"


def test_linked_attributes_can_join_on_transaction_ids(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle_1.zip",
        ucl.LINKED_MEMBER,
        {"tranall_link_02122023.csv": _linked_csv()},
    )
    # Price Paid prints ids in braces and upper case.
    ppd = tmp_path / "pp-2019.csv"
    ppd.write_text(
        '"{aaaa-0001}","1","2010-03-05 00:00","AB1 2CD","D","N","F","1","","S","","T","D","C","A","A"\n'
        '"{AAAA-0002}","1","2019-06-01 00:00","AB1 2CD","D","N","F","1","","S","","T","D","C","A","A"\n',
        encoding="utf-8",
    )
    keys = ucl.price_paid_transaction_keys(ppd, key_function=hv.dwelling_key)
    no_address = ucl.LinkedColumns(
        transaction_id="transactionid",
        date="dateoftransfer",
        floor_area="tfarea",
        age_band="CONSTRUCTION_AGE_BAND",
    )
    out, receipts = ucl.linked_attributes(
        bundle,
        no_address,
        age_bands=_BANDS,
        floor_area_bounds=_BOUNDS,
        key_function=None,
        transaction_keys=keys,
    )
    assert out["key"].tolist() == [_key("AB12CD", "1", "")]
    assert receipts["rows"] == 4 and receipts["rows_joined"] == 2
    assert receipts["join"] == "transaction id"


def test_a_declared_column_the_file_lacks_is_refused(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "bundle_1.zip",
        ucl.LINKED_MEMBER,
        {"tranall_link_02122023.csv": _linked_csv()},
    )
    renamed = ucl.LinkedColumns(
        **{**_COLUMNS.__dict__, "floor_area": "TOTAL_FLOOR_AREA"}
    )
    with pytest.raises(ValueError, match="discover"):
        ucl.linked_attributes(
            bundle,
            renamed,
            age_bands=_BANDS,
            floor_area_bounds=_BOUNDS,
            key_function=hv.dwelling_key,
        )


def test_readme_text_reads_plain_text_and_word(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle_1.zip", ucl.LINKED_MEMBER, {"x.csv": b"a\n1\n"})
    name, text, digest = ucl.readme_text(bundle)
    assert name == "856542_readme.txt" and "tfarea" in text and len(digest) == 64
    document = io.BytesIO()
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(document, "w") as docx:
        docx.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{w}"><w:body>'
            "<w:p><w:r><w:t>Field list</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>tfarea: total floor area</w:t></w:r></w:p>"
            "</w:body></w:document>",
        )
    word = tmp_path / "bundle_word.zip"
    with zipfile.ZipFile(word, "w") as outer:
        outer.writestr("ReadMe.docx", document.getvalue())
    assert ucl.readme_text(word)[1] == "Field list\ntfarea: total floor area"
    members = ucl.bundle_members(bundle)
    assert {m["name"]: m["stored"] for m in members}[ucl.LINKED_MEMBER] is True


def test_age_bands_and_dwelling_types_read_the_epc_labels() -> None:
    labels = pd.Series(
        [
            "England and Wales: before 1900",
            "England and Wales: 1967-1975",
            "1983-1990",
            "2007 onwards",
            "2021",
            "NO DATA!",
            None,
        ]
    )
    assert ucl.age_band_index(labels, _BANDS).tolist() == [0, 4, 6, 10, 11, -1, -1]
    kinds = pd.Series(["House", "Flat", "Bungalow", "House", "Park home", "House"])
    forms = pd.Series(
        [
            "Detached",
            "Mid-Terrace",
            "Semi-Detached",
            "Enclosed End-Terrace",
            "Detached",
            None,
        ]
    )
    assert ucl.dwelling_type_index(kinds, forms).tolist() == [0, 3, 1, 2, -1, -1]


_CERTIFICATE_HEADER = [
    "LMK_KEY",
    "LOCAL_AUTHORITY",
    "TENURE",
    "PROPERTY_TYPE",
    "BUILT_FORM",
    "TOTAL_FLOOR_AREA",
    "CONSTRUCTION_AGE_BAND",
    "LODGEMENT_DATE",
    "UPRN",
]


def _certificates(rows: list[list[str]]) -> bytes:
    lines = [",".join(_CERTIFICATE_HEADER)] + [",".join(row) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_social_certificate_cells_count_each_building_once(tmp_path: Path) -> None:
    first = _certificates(
        [
            # One building certified twice: only the later certificate counts.
            [
                "1",
                "E08000038",
                "rental (social)",
                "Flat",
                "Mid-Terrace",
                "50",
                "1967-1975",
                "2012-01-01",
                "100",
            ],
            [
                "2",
                "E08000038",
                "Rented (social)",
                "Flat",
                "Mid-Terrace",
                "60",
                "1967-1975",
                "2018-05-01",
                "100",
            ],
            [
                "3",
                "E08000038",
                "owner-occupied",
                "House",
                "Detached",
                "120",
                "1930-1949",
                "2015-01-01",
                "101",
            ],
            [
                "4",
                "E08000038",
                "RENTAL (SOCIAL)",
                "House",
                "Semi-Detached",
                "",
                "1950-1966",
                "2016-01-01",
                "102",
            ],
        ]
    )
    second = _certificates(
        [
            [
                "5",
                "W06000001",
                "rental (social)",
                "Park home",
                "Detached",
                "40",
                "1983-1990",
                "2014-01-01",
                "200",
            ],
            [
                "6",
                "W06000001",
                "rental (social)",
                "House",
                "End-Terrace",
                "70",
                "1983-1990",
                "2014-01-01",
                "201",
            ],
        ]
    )
    bundle = _bundle(
        tmp_path / "bundle_2.zip",
        ucl.EPC_MEMBER,
        {"epc/a.csv": first, "epc/b.csv": second},
    )
    columns = ucl.CertificateColumns(**declared.UCL_CERTIFICATE_COLUMNS)
    cells, receipts = ucl.social_certificate_cells(
        bundle,
        columns,
        age_bands=_BANDS,
        floor_area_bounds=_BOUNDS,
        authority_aliases={"E08000038": "E08000016"},
    )
    assert cells.to_dict("list") == {
        "local_authority": ["E08000016", "E08000016", "W06000001"],
        "ptype": [1, 3, 2],
        "age_band": [3, 4, 6],
        "certificates": [1, 1, 1],
        "floor_area_mean": [pytest.approx(np.nan, nan_ok=True), 60.0, 70.0],
    }
    assert receipts["certificates_read"] == 6
    assert receipts["social_certificates"] == 5
    assert receipts["social_buildings"] == 4
    assert receipts["social_without_type_or_age"] == 1  # the park home
    assert receipts["members"] == 2
