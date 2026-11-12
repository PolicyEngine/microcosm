"""UCL's Price Paid–EPC linked file (ReShare 856542), read as a stream.

Chi, Dennett, Oléron-Evans and Morphet (UCL) linked HM Land Registry's Price
Paid sales in England and Wales from 1995 to October 2023 to the domestic
Energy Performance Certificates, which carry each home's floor area, built
form and construction age band. Chronicle registers both bundles raw-only
(PolicyEngine/chronicle#426):

- **Bundle 1** holds the linked file, ``tranall_link_02122023.zip``, inside
  the bundle zip, with UCL's readme. :func:`linked_attributes` streams its
  CSV from the inner zip without extracting anything to disk, reads only
  the declared columns, and keeps each dwelling's latest linked sale with a
  usable value for each attribute: so a home sold after October 2023 takes
  an earlier linked sale's floor area and age.
- **Bundle 2** holds the certificates UCL linked, with addresses removed
  (``epc20_id.zip``). :func:`social_certificate_cells` counts the social
  rented ones by authority, type and age band: the stock cell builder's
  floor areas and ages for social rented homes, which never sell.

The column names are a declaration (:class:`LinkedColumns`,
:class:`CertificateColumns`), checked against each file's header row before
any row is read. UCL's readme names them; the builder's ``discover`` stage
prints the header so the declaration can be checked against it.

The files hold Royal Mail address data for non-commercial use. Nothing
derived here keeps an address: a dwelling is its uint64 key, which the
caller's key function (the home-stock library's ``dwelling_key``) computes.
This module is loaded by file path from the builder tool, so it imports no
``uk_runtime`` sibling.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import re
import struct
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv as pacsv

__all__ = [
    "CertificateColumns",
    "EPC_MEMBER",
    "LINKED_MEMBER",
    "LinkedColumns",
    "age_band_index",
    "bundle_members",
    "csv_header",
    "dwelling_type_index",
    "iter_csv_batches",
    "linked_attributes",
    "open_inner_zip",
    "price_paid_transaction_keys",
    "readme_text",
    "social_certificate_cells",
    "transaction_hash",
]

LINKED_MEMBER = "tranall_link_02122023.zip"
EPC_MEMBER = "epc20_id.zip"
_README = re.compile(r"read[\s_-]*me", re.IGNORECASE)
#: A zip local file header: signature, versions, flags, method, times, CRC,
#: sizes, then the lengths of the file name and the extra field.
_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")
_LOCAL_SIGNATURE = b"PK\x03\x04"
_TEXT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_PARAGRAPH = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
#: Price Paid's four types, in the home-stock library's ``ptype`` order.
_TYPES = ("D", "S", "T", "F")


@dataclass(frozen=True)
class LinkedColumns:
    """The linked file's columns the builder reads, by role.

    ``postcode``, ``paon`` and ``saon`` rebuild the dwelling key. Without all
    three, the caller joins on the transaction id instead
    (:func:`price_paid_transaction_keys`).
    """

    transaction_id: str
    date: str
    floor_area: str
    age_band: str
    postcode: str | None = None
    paon: str | None = None
    saon: str | None = None
    encoding: str = "utf8"

    @property
    def has_address(self) -> bool:
        return None not in (self.postcode, self.paon, self.saon)

    def names(self) -> tuple[str, ...]:
        roles = (
            self.transaction_id,
            self.date,
            self.floor_area,
            self.age_band,
            self.postcode,
            self.paon,
            self.saon,
        )
        return tuple(dict.fromkeys(name for name in roles if name is not None))


@dataclass(frozen=True)
class CertificateColumns:
    """The certificate file's columns the builder reads, by role."""

    local_authority: str
    tenure: str
    property_type: str
    built_form: str
    floor_area: str
    age_band: str
    lodgement_date: str
    social_tenures: tuple[str, ...]
    building_id: str | None = None
    encoding: str = "utf8"

    def names(self) -> tuple[str, ...]:
        roles = (
            self.local_authority,
            self.tenure,
            self.property_type,
            self.built_form,
            self.floor_area,
            self.age_band,
            self.lodgement_date,
            self.building_id,
        )
        return tuple(dict.fromkeys(name for name in roles if name is not None))


# ----------------------------------------------------------------------------
# Zips inside zips


class _Window(io.RawIOBase):
    """A read-only, seekable view of ``size`` bytes of a file from ``start``."""

    def __init__(self, handle, start: int, size: int) -> None:
        self._handle = handle
        self._start = start
        self._size = size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}
        self._position = max(0, min(self._size, base[whence] + offset))
        return self._position

    def readinto(self, buffer) -> int:
        view = memoryview(buffer)
        wanted = min(len(view), self._size - self._position)
        if wanted <= 0:
            return 0
        self._handle.seek(self._start + self._position)
        read = self._handle.readinto(view[:wanted])
        self._position += read
        return read


def bundle_members(path: Path) -> list[dict[str, object]]:
    """Each member of a zip: its name, size and whether it is stored as is."""
    with zipfile.ZipFile(path) as archive:
        return [
            {
                "name": info.filename,
                "size_bytes": info.file_size,
                "stored": info.compress_type == zipfile.ZIP_STORED,
            }
            for info in archive.infolist()
        ]


def _member(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    """A member by its name, or by its file name inside one folder.

    ReShare's bundles hold their files in a folder named for the bundle
    (``856542_bundle_1/tranall_link_02122023.zip``).
    """
    try:
        return archive.getinfo(name)
    except KeyError:
        matches = [
            info for info in archive.infolist() if Path(info.filename).name == name
        ]
        if len(matches) != 1:
            raise KeyError(
                f"The bundle holds {len(matches)} members named {name!r}."
            ) from None
        return matches[0]


@contextlib.contextmanager
def open_inner_zip(bundle_path: Path, member: str) -> Iterator[zipfile.ZipFile]:
    """Open a zip that is itself a member of the bundle zip, without extracting it.

    A stored member is read in place through a window on the bundle file, so
    seeking in it costs nothing; a compressed one falls back to zipfile's own
    stream, which can seek only by reading.
    """
    with open(bundle_path, "rb") as handle, zipfile.ZipFile(handle) as outer:
        info = _member(outer, member)
        if info.compress_type == zipfile.ZIP_STORED:
            handle.seek(info.header_offset)
            fields = _LOCAL_HEADER.unpack(handle.read(_LOCAL_HEADER.size))
            if fields[0] != _LOCAL_SIGNATURE:
                raise ValueError(f"{member}: bad local file header in {bundle_path}.")
            start = info.header_offset + _LOCAL_HEADER.size + fields[10] + fields[11]
            stream = io.BufferedReader(
                _Window(handle, start, info.compress_size), buffer_size=1 << 20
            )
        else:
            stream = outer.open(member)
        with stream, zipfile.ZipFile(stream) as inner:
            yield inner


def _csv_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        info.filename
        for info in archive.infolist()
        if info.filename.lower().endswith(".csv") and not info.is_dir()
    )


def csv_header(archive: zipfile.ZipFile, member: str, *, encoding: str) -> list[str]:
    """A CSV member's header row, read without the rest (a byte-order mark dropped)."""
    codec = "utf-8-sig" if encoding.replace("-", "").lower() == "utf8" else encoding
    with archive.open(member) as stream:
        text = io.TextIOWrapper(stream, encoding=codec, newline="")
        return next(csv.reader(text))


def _require_columns(header: Sequence[str], wanted: Sequence[str], where: str) -> None:
    missing = [name for name in wanted if name not in header]
    if missing:
        raise ValueError(
            f"{where} lacks the declared columns {missing}; run the discover "
            "stage and correct the declaration."
        )


def iter_csv_batches(
    stream,
    columns: Sequence[str],
    *,
    header: Sequence[str],
    encoding: str,
    block_size: int = 1 << 24,
) -> Iterator[pa.RecordBatch]:
    """Record batches of the named columns, every value as text.

    The header row is skipped and ``header`` names the columns, so a
    byte-order mark before the first name cannot hide it.
    """
    reader = pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(
            block_size=block_size,
            encoding=encoding,
            column_names=list(header),
            skip_rows=1,
        ),
        convert_options=pacsv.ConvertOptions(
            include_columns=list(columns),
            column_types={name: pa.string() for name in columns},
            strings_can_be_null=True,
        ),
    )
    yield from reader


def readme_text(bundle_path: Path) -> tuple[str, str | None, str]:
    """UCL's readme in the bundle: its name, its text when it has text, its sha256.

    A ``.docx`` readme is read from its document XML; a plain-text one is
    decoded; any other kind (a PDF, say) returns no text.
    """
    with zipfile.ZipFile(bundle_path) as archive:
        names = [name for name in archive.namelist() if _README.search(Path(name).name)]
        if len(names) != 1:
            raise ValueError(
                f"{bundle_path} holds {len(names)} readme members: {names}"
            )
        (name,) = names
        payload = archive.read(name)
    digest = hashlib.sha256(payload).hexdigest()
    suffix = Path(name).suffix.lower()
    if suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(payload)) as document:
            root = ElementTree.fromstring(document.read("word/document.xml"))
        paragraphs = [
            "".join(node.text or "" for node in paragraph.iter(_TEXT))
            for paragraph in root.iter(_PARAGRAPH)
        ]
        return name, "\n".join(paragraphs), digest
    if suffix in (".txt", ".md", ".csv", ""):
        return name, payload.decode("utf-8", errors="replace"), digest
    return name, None, digest


# ----------------------------------------------------------------------------
# Values


def age_band_index(
    values: pd.Series, bands: Sequence[tuple[str, int, int]]
) -> np.ndarray:
    """Each EPC construction age band's position in ``bands``, or -1.

    ``bands`` holds (label, first year, last year). The EPC writes a band as
    "before 1900", "1930-1949", "2012 onwards" (often after "England and
    Wales: "), or a single year on a new-build certificate. A range takes the
    band that holds its middle year, "onwards" the band that holds its first
    year, and "before" the first band. Anything else is -1.
    """
    text = values.astype("string").str.strip().str.lower()
    codes, uniques = pd.factorize(text, use_na_sentinel=True)

    def place(label: str) -> int:
        years = [int(y) for y in re.findall(r"(?<!\d)(\d{4})(?!\d)", label)]
        if "before" in label and years:
            year = years[0] - 1
        elif len(years) >= 2:
            year = (years[0] + years[1]) // 2
        elif len(years) == 1:
            year = years[0]
        else:
            return -1
        for position, (_, first, last) in enumerate(bands):
            if first <= year <= last:
                return position
        # The open bands at either end hold any year beyond them.
        return 0 if year < bands[0][1] else len(bands) - 1

    lookup = np.array([place(str(label)) for label in uniques] + [-1], dtype=np.int8)
    return lookup[np.where(codes < 0, len(uniques), codes)]


def dwelling_type_index(property_type: pd.Series, built_form: pd.Series) -> np.ndarray:
    """Price Paid's type (0 D, 1 S, 2 T, 3 F) from an EPC's type and built form.

    Flats and maisonettes are F. Houses and bungalows take their built form:
    detached, semi-detached, or any terrace. A park home, or a house with no
    usable built form, is -1.
    """
    kind = property_type.astype("string").str.strip().str.lower().fillna("")
    form = built_form.astype("string").str.strip().str.lower().fillna("")
    out = np.full(len(kind), -1, dtype=np.int8)
    flat = kind.isin(["flat", "maisonette"]).to_numpy()
    house = kind.isin(["house", "bungalow"]).to_numpy()
    out[flat] = 3
    out[house & (form == "detached").to_numpy()] = 0
    out[house & (form == "semi-detached").to_numpy()] = 1
    out[house & form.str.contains("terrace", regex=False).to_numpy()] = 2
    return out


def _yyyymm(values: pd.Series) -> np.ndarray:
    """A date's year × 100 + month, from "2015-03-05 ..." or "05/03/2015"; else 0."""
    text = values.astype("string").str.strip()
    iso = text.str.extract(r"^(?P<year>\d{4})-(?P<month>\d{1,2})")
    british = text.str.extract(r"^\d{1,2}/(?P<month>\d{1,2})/(?P<year>\d{4})")
    year = pd.to_numeric(iso["year"].fillna(british["year"]), errors="coerce")
    month = pd.to_numeric(iso["month"].fillna(british["month"]), errors="coerce")
    ok = (year.notna() & month.between(1, 12)).to_numpy(dtype=bool, na_value=False)
    out = np.zeros(len(text), dtype=np.int32)
    out[ok] = (year[ok] * 100 + month[ok]).to_numpy(dtype=np.int64)
    return out


def _floor_area(values: pd.Series, bounds: tuple[float, float]) -> np.ndarray:
    area = pd.to_numeric(values.astype("string").str.strip(), errors="coerce")
    area = area.to_numpy(dtype=float, na_value=np.nan)
    low, high = bounds
    return np.where((area >= low) & (area <= high), area, np.nan).astype(np.float32)


def transaction_hash(values: pd.Series) -> np.ndarray:
    """A uint64 hash of Price Paid transaction ids, braces and case dropped."""
    text = values.astype("string").str.strip().str.strip("{}").str.upper().fillna("")
    return pd.util.hash_pandas_object(text, index=False).to_numpy(dtype=np.uint64)


def _latest(frame: pd.DataFrame, column: str, valid: np.ndarray) -> pd.DataFrame:
    rows = frame.loc[valid, ["key", "yyyymm", column]]
    rows = rows.sort_values(["key", "yyyymm"], kind="stable")
    return rows.drop_duplicates("key", keep="last").rename(
        columns={"yyyymm": f"{column}_yyyymm"}
    )


# ----------------------------------------------------------------------------
# The linked file


def linked_attributes(
    bundle_path: Path,
    columns: LinkedColumns,
    *,
    age_bands: Sequence[tuple[str, int, int]],
    floor_area_bounds: tuple[float, float],
    key_function: Callable[[pd.Series, pd.Series, pd.Series], np.ndarray] | None,
    transaction_keys: pd.DataFrame | None = None,
    block_size: int = 1 << 24,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Each linked dwelling's floor area and age band, from its latest linked sale.

    The dwelling key is rebuilt from the file's postcode, PAON and SAON with
    ``key_function`` when the declaration has all three; otherwise each
    sale's transaction id is looked up in ``transaction_keys`` (columns
    ``transaction`` and ``key``, from :func:`price_paid_transaction_keys`).
    Each attribute comes from the dwelling's latest sale that carries a
    usable value of it. Returns columns ``key``, ``floor_area``,
    ``floor_area_yyyymm``, ``age_band`` and ``age_band_yyyymm`` (-1 or NaN
    where the dwelling has none), sorted by key, and the receipts.
    """
    if columns.has_address and key_function is None:
        raise ValueError("A key function is needed to rebuild the dwelling key.")
    if not columns.has_address and transaction_keys is None:
        raise ValueError("Without an address, the join needs the transaction keys.")
    names = columns.names()
    parts: list[pd.DataFrame] = []
    receipts: dict[str, object] = {"rows": 0, "batches": 0}
    with open_inner_zip(bundle_path, LINKED_MEMBER) as inner:
        members = _csv_members(inner)
        if len(members) != 1:
            raise ValueError(f"The linked zip holds {len(members)} CSV members.")
        (member,) = members
        header = csv_header(inner, member, encoding=columns.encoding)
        _require_columns(header, names, member)
        if transaction_keys is not None:
            transaction_keys = transaction_keys.drop_duplicates("transaction")
        with inner.open(member) as stream:
            for batch in iter_csv_batches(
                stream,
                names,
                header=header,
                encoding=columns.encoding,
                block_size=block_size,
            ):
                frame = batch.to_pandas()
                if columns.has_address:
                    key = key_function(
                        frame[columns.postcode],
                        frame[columns.paon],
                        frame[columns.saon],
                    )
                    joined = np.ones(len(frame), dtype=bool)
                else:
                    position = pd.Index(transaction_keys["transaction"]).get_indexer(
                        transaction_hash(frame[columns.transaction_id])
                    )
                    joined = position >= 0
                    key = np.where(
                        joined,
                        transaction_keys["key"].to_numpy()[np.maximum(position, 0)],
                        0,
                    ).astype(np.uint64)
                parts.append(
                    pd.DataFrame(
                        {
                            "key": key[joined],
                            "yyyymm": _yyyymm(frame[columns.date])[joined],
                            "floor_area": _floor_area(
                                frame[columns.floor_area], floor_area_bounds
                            )[joined],
                            "age_band": age_band_index(
                                frame[columns.age_band], age_bands
                            )[joined],
                        }
                    )
                )
                receipts["rows"] += len(frame)
                receipts["batches"] += 1
                if log is not None and receipts["batches"] % 20 == 0:
                    log(f"  {receipts['rows']:,} linked sales read")
    frame = pd.concat(parts, ignore_index=True)
    del parts
    receipts["rows_joined"] = int(len(frame))
    floor = _latest(frame, "floor_area", np.isfinite(frame["floor_area"].to_numpy()))
    age = _latest(frame, "age_band", frame["age_band"].to_numpy() >= 0)
    out = floor.merge(age, on="key", how="outer").sort_values("key", kind="stable")
    out["floor_area"] = out["floor_area"].astype(np.float32)
    out["age_band"] = out["age_band"].fillna(-1).astype(np.int8)
    for column in ("floor_area_yyyymm", "age_band_yyyymm"):
        out[column] = out[column].fillna(0).astype(np.int32)
    receipts.update(
        {
            "rows_with_floor_area": int(np.isfinite(frame["floor_area"]).sum()),
            "rows_with_age_band": int((frame["age_band"] >= 0).sum()),
            "dwellings": int(len(out)),
            "dwellings_with_floor_area": int(len(floor)),
            "dwellings_with_age_band": int(len(age)),
            "join": "address key" if columns.has_address else "transaction id",
            "columns": names,
        }
    )
    return out.reset_index(drop=True), receipts


def price_paid_transaction_keys(
    path: Path,
    *,
    key_function: Callable[[pd.Series, pd.Series, pd.Series], np.ndarray],
) -> pd.DataFrame:
    """Each Price Paid transaction's hashed id and its dwelling key, for one year.

    Price Paid's field order is its contract: the id is field 0, the postcode
    3, PAON 7 and SAON 8. Only the join on transaction ids needs this.
    """
    table = pd.read_csv(
        path,
        header=None,
        usecols=[0, 3, 7, 8],
        names=["transaction", "postcode", "paon", "saon"],
        dtype="string",
        engine="pyarrow",
    )
    return pd.DataFrame(
        {
            "transaction": transaction_hash(table["transaction"]),
            "key": key_function(table["postcode"], table["paon"], table["saon"]),
        }
    )


# ----------------------------------------------------------------------------
# Social rented certificates


def social_certificate_cells(
    bundle_path: Path,
    columns: CertificateColumns,
    *,
    age_bands: Sequence[tuple[str, int, int]],
    floor_area_bounds: tuple[float, float],
    authority_aliases: Mapping[str, str] | None = None,
    block_size: int = 1 << 24,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Social rented certificates by authority, type and age band.

    A certificate counts when its tenure is one of the declared social
    tenures (case and spaces aside). With a building id, only each
    building's latest certificate counts, so a home certified twice counts
    once. Returns columns ``local_authority``, ``ptype``, ``age_band``,
    ``certificates`` and ``floor_area_mean`` (over certificates with a usable
    floor area; NaN where none has one), and the receipts.
    """
    names = columns.names()
    social = {value.strip().lower() for value in columns.social_tenures}
    aliases = dict(authority_aliases or {})
    parts: list[pd.DataFrame] = []
    receipts: dict[str, object] = {"certificates_read": 0, "members": 0}
    with open_inner_zip(bundle_path, EPC_MEMBER) as inner:
        members = _csv_members(inner)
        if not members:
            raise ValueError("The certificate zip holds no CSV member.")
        for member in members:
            header = csv_header(inner, member, encoding=columns.encoding)
            _require_columns(header, names, member)
            with inner.open(member) as stream:
                for batch in iter_csv_batches(
                    stream,
                    names,
                    header=header,
                    encoding=columns.encoding,
                    block_size=block_size,
                ):
                    frame = batch.to_pandas()
                    receipts["certificates_read"] += len(frame)
                    tenure = frame[columns.tenure].astype("string").str.strip()
                    keep = tenure.str.lower().isin(social).fillna(False).to_numpy()
                    frame = frame[keep]
                    if frame.empty:
                        continue
                    authority = (
                        frame[columns.local_authority].astype("string").str.strip()
                    )
                    parts.append(
                        pd.DataFrame(
                            {
                                "local_authority": authority.map(
                                    lambda code: aliases.get(code, code)
                                ).astype(str),
                                "building": (
                                    transaction_hash(frame[columns.building_id])
                                    if columns.building_id
                                    else np.arange(len(frame), dtype=np.uint64)
                                ),
                                "lodged": _yyyymm(frame[columns.lodgement_date]),
                                "ptype": dwelling_type_index(
                                    frame[columns.property_type],
                                    frame[columns.built_form],
                                ),
                                "age_band": age_band_index(
                                    frame[columns.age_band], age_bands
                                ),
                                "floor_area": _floor_area(
                                    frame[columns.floor_area], floor_area_bounds
                                ),
                            }
                        )
                    )
            receipts["members"] += 1
            if log is not None:
                log(f"  {member}: {receipts['certificates_read']:,} certificates read")
    if not parts:
        raise ValueError("No certificate carries a declared social tenure.")
    frame = pd.concat(parts, ignore_index=True)
    receipts["social_certificates"] = int(len(frame))
    if columns.building_id:
        frame = frame.sort_values(["building", "lodged"], kind="stable")
        frame = frame.drop_duplicates("building", keep="last")
    receipts["social_buildings"] = int(len(frame))
    usable = (frame["ptype"] >= 0) & (frame["age_band"] >= 0)
    receipts["social_without_type_or_age"] = int((~usable).sum())
    frame = frame[usable].copy()
    frame["has_area"] = np.isfinite(frame["floor_area"].to_numpy())
    frame["area"] = frame["floor_area"].fillna(0.0).astype(float)
    cells = (
        frame.groupby(["local_authority", "ptype", "age_band"], sort=True)
        .agg(
            certificates=("ptype", "size"),
            with_area=("has_area", "sum"),
            area_total=("area", "sum"),
        )
        .reset_index()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        cells["floor_area_mean"] = np.where(
            cells["with_area"] > 0, cells["area_total"] / cells["with_area"], np.nan
        )
    receipts["deduplicated_by"] = columns.building_id or "none"
    out = cells[
        ["local_authority", "ptype", "age_band", "certificates", "floor_area_mean"]
    ]
    out = out.astype({"ptype": np.int8, "age_band": np.int8, "certificates": np.int64})
    return out.reset_index(drop=True), receipts
