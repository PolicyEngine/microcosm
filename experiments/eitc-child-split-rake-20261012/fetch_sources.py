"""Download the SOI files ``sources.json`` pins, verify them, dump them to TSV.

Each workbook sheet becomes ``<file>.tsv`` (one line per row, the row number
first, cells tab-separated, newlines inside cells flattened) for
``eitc_core.py`` to parse. A file whose SHA-256 differs from the pin is
refused.

Run: uv run --no-project --with xlrd==2.0.1 --with openpyxl==3.1.5 \
       python -I fetch_sources.py <download_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import openpyxl
import xlrd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "tsv").mkdir(exist_ok=True)

for source in json.loads((HERE / "sources.json").read_text())["sources"]:
    path = OUT / source["file"]
    if not path.exists():
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request) as response:  # noqa: S310 (pinned irs.gov URLs)
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise SystemExit(f"{path.name}: sha256 {digest} != pinned {source['sha256']}")
    lines = []
    if path.suffix == ".xlsx":
        book = openpyxl.load_workbook(path, data_only=True, read_only=True)
        for sheet in book.worksheets:
            lines.append(f"### sheet {sheet.title}")
            for r, row in enumerate(sheet.iter_rows(values_only=True)):
                cells = ("" if v is None else str(v).replace("\n", " ") for v in row)
                lines.append(f"{r}\t" + "\t".join(cells))
    elif path.suffix == ".xls":
        book = xlrd.open_workbook(path)
        for sheet in book.sheets():
            lines.append(f"### sheet {sheet.name}")
            for r in range(sheet.nrows):
                cells = (str(v).replace("\n", " ") for v in sheet.row_values(r))
                lines.append(f"{r}\t" + "\t".join(cells))
    else:
        print(f"{path.name}: verified")
        continue
    (OUT / "tsv" / f"{path.name}.tsv").write_text("\n".join(lines))
    print(f"{path.name}: verified, {len(lines)} lines")
