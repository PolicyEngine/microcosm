"""Closed PAW source observations, independent of the sealed 33-field money body.

Only exact source reconstruction issues this authority. Candidate replay compares
canonical expected bytes before parsing any candidate. No Frame columns are added.
"""

from __future__ import annotations

import csv
import hashlib
import os
import stat
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path

import numpy as np
import pandas as pd

from . import asec_current_money_source as legacy
from . import asec_person_income_source as restoration
from ._asec_current_money_codec import current_money_content_sha256
from .asec_current_money import (
    RESOURCE_PINS,
    RESTORED_SOURCE_KIND,
    MoneyRefusalError,
    ReadyCurrentMoney,
    _json,
    _parse,
    _require,
    _sha,
    _validate_money,
)
from .asec_student_controls import _snapshot
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

MAGIC = b"MCAINCOB\x01"
FILENAME = "asec_income_observations.bin"
ARTIFACT_KIND = "microcosm.us.asec_income_observations.v1"
_HEADER_MAX = 65536
_MAX_PERSONS = 600_000
INCOME_PAYLOAD_MAX_BYTES = len(MAGIC) + 4 + _HEADER_MAX + 8 * 8 * _MAX_PERSONS + 32
COORDINATES = ("person_id", "income_year", "source_household_id", "A_LINENO", "A_AGE")
COLUMNS = COORDINATES + ("PAW_YN", "PAW_VAL", "PAW_VAL_2024_price")
COLUMN_DTYPES = {
    name: "little_endian_float64" if name == COLUMNS[-1] else "little_endian_int64"
    for name in COLUMNS
}
_READ_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE", "PAW_VAL", "PAW_YN")
_MEMBER_PINS = tuple(
    (year, p.member, p.zip_sha256, p.member_sha256, p.rows, p.member_size_bytes)
    for year, p in sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items())
)
RESOURCE = "asec_income_observations_v1.json"
RESOURCE_SHA256 = "6e37855ddd6642d02231efc664644029ed6131885eb743a00197261bc5a6a51f"
ZERO_CLASSES = (
    "observed_amount",
    "niu_below_receipt_age",
    "reported_receipt_zero_amount",
    "reported_no_receipt",
    "niu_in_age_universe",
)
_TOKEN = object()
_CSV_DIALECT = "excel"
_CSV_ENCODING = "utf-8"
_PRICE_TARGET = "174.4"
_PRICE_YEARS = (2022, 2023, 2024)


def parser_profile() -> dict:
    """Read the effective CSV parser limits and dialect, without changing them.

    ``csv.field_size_limit`` is process state that silently changes which member
    files parse. Binding it into the source identity makes any change to it a
    different artifact identity instead of an invisible reparse.
    """
    # _read_member uses csv.reader's built-in default, which is independent of
    # the mutable named-dialect registry. Inspect that same reader's dialect.
    dialect = csv.reader(()).dialect
    return {
        "header_engine": "stdlib_csv_reader",
        "table_engine": "pandas_read_csv",
        "dialect": _CSV_DIALECT,
        "encoding": _CSV_ENCODING,
        "newline": "",
        "delimiter": dialect.delimiter,
        "quotechar": dialect.quotechar,
        "doublequote": bool(dialect.doublequote),
        "skipinitialspace": bool(dialect.skipinitialspace),
        "strict": bool(dialect.strict),
        "quoting": int(dialect.quoting),
        "csv.field_size_limit": csv.field_size_limit(),
    }


def price_factors(contract) -> tuple[tuple[int, str, str], ...]:
    """The single restatement-factor derivation, shared with graph transport."""
    return tuple(
        (
            year,
            f"{_PRICE_TARGET}/{contract['price']['cells'][str(year)]}",
            struct.pack(
                "<d",
                float(_PRICE_TARGET) / float(contract["price"]["cells"][str(year)]),
            ).hex(),
        )
        for year in _PRICE_YEARS
    )


_HEADER_KEYS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "rows",
        "columns",
        "column_dtypes",
        "source_identity",
        "t1_sha256",
        "source_frame_sha256",
        "scope_sha256",
        "native_person_keys_sha256",
        "money_header_sha256",
        "money_content_sha256",
        "sources",
        "join_keys",
        "crosschecks",
        "price",
        "monetary_basis",
        "zero_origin_derivation",
        "reference_period",
        "release_eligible",
        "implementation",
        "body_sha256",
        "column_sha256",
        "encoding_contract",
    }
)


def _contract():
    payload = resources.files(__package__).joinpath(RESOURCE).read_bytes()
    _require(_sha(payload) == RESOURCE_SHA256, "INCOME_RESOURCE_BYTES")
    return _parse(payload)


def _implementation():
    package = resources.files(__package__)
    _contract()
    return {
        "schema": 1,
        "t1_verification_sha256": restoration._implementation(),
        "modules": {
            name: _sha(package.joinpath(name).read_bytes())
            for name in (
                "asec_income_observations.py",
                "asec_student_controls.py",
                "_asec_current_money_codec.py",
            )
        },
        "dependencies": {n: metadata.version(n) for n in ("numpy", "pandas")},
        "member_pins": _MEMBER_PINS,
        "resource_sha256": RESOURCE_SHA256,
        "parser_profile": parser_profile(),
    }


def _parent(source, ready, *, reconstruct_money=False):
    _require(
        type(source) is legacy.AuthenticatedCurrentMoneySource
        and type(ready) is ReadyCurrentMoney,
        "INCOME_AUTHENTICATED_PARENT",
    )
    source.validate()
    _require(
        _parse(source.source.identity)["source_kind"] == RESTORED_SOURCE_KIND,
        "INCOME_RESTORED_PARENT_REQUIRED",
    )
    _require(ready.bindings.spec == source.spec, "INCOME_MONEY_PARENT")
    _validate_money(ready, source.spec, nominal=False)
    if reconstruct_money:
        expected = source.ready()
        _require(
            ready.bindings == expected.bindings and ready.fields == expected.fields,
            "INCOME_MONEY_PARENT",
        )


def _capture(path, destination, *, size):
    """Use S's bounded descriptor capture, with a regular path identity check."""
    before = Path(path).lstat()
    _require(stat.S_ISREG(before.st_mode), "INCOME_SOURCE_FILE_KIND")
    digest = _snapshot(path, destination, size=size)
    after = Path(path).lstat()
    attrs = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        stat.S_ISREG(after.st_mode)
        and all(getattr(before, a) == getattr(after, a) for a in attrs),
        "INCOME_SOURCE_CHANGED",
    )
    return digest


def _read_member(path, *, rows):
    with path.open("r", encoding=_CSV_ENCODING, newline="") as handle:
        header = next(csv.reader(handle), [])
    _require(
        len(header) == len(set(header)) and set(_READ_COLUMNS) <= set(header),
        "INCOME_MEMBER_HEADER",
    )
    table = pd.read_csv(
        path, usecols=list(_READ_COLUMNS), dtype="string", na_filter=False
    )
    _require(len(table) == rows, "INCOME_MEMBER_ROWS")
    _require(
        bool(table.PERIDNUM.str.fullmatch(r"[0-9]{22}").all()),
        "INCOME_MEMBER_PERSON_KEY",
    )
    for name in _READ_COLUMNS[1:]:
        _require(
            bool(table[name].str.fullmatch(r"[0-9]+").all()),
            "INCOME_MEMBER_INTEGER",
            name,
        )
        table[name] = table[name].astype("int64")
    _require(
        not table.PERIDNUM.duplicated().any()
        and not table.duplicated(["PH_SEQ", "A_LINENO"]).any()
        and bool((table.PH_SEQ > 0).all())
        and bool((table.A_LINENO > 0).all()),
        "INCOME_MEMBER_COORDINATES",
    )
    _validate_observations(
        table.A_AGE.to_numpy(), table.PAW_YN.to_numpy(), table.PAW_VAL.to_numpy()
    )
    return table


def _validate_observations(
    age: np.ndarray, receipt_code: np.ndarray, nominal: np.ndarray
):
    _require(
        all(
            type(a) is np.ndarray
            and a.dtype == np.dtype("int64")
            and a.ndim == 1
            and a.shape == nominal.shape
            for a in (age, receipt_code, nominal)
        ),
        "INCOME_OBSERVATION_ARRAYS",
    )
    _require(
        bool(((nominal >= 0) & (nominal <= 99999)).all()),
        "INCOME_CODE_DOMAIN",
        "PAW_VAL",
    )
    _require(
        bool(np.isin(receipt_code, [0, 1, 2]).all()), "INCOME_CODE_DOMAIN", "PAW_YN"
    )
    _require(
        bool((receipt_code[nominal != 0] == 1).all())
        and bool((nominal[receipt_code != 1] == 0).all())
        and bool((receipt_code[age < 15] == 0).all())
        and bool((nominal[age < 15] == 0).all()),
        "INCOME_SOURCE_UNIVERSE",
    )


def _restate_amounts(nominal, years, factors):
    """The single PAW restatement path, also used to verify graph transport."""
    amounts = nominal.astype("<f8")
    for year, _ratio, bits in factors:
        if year != 2024:
            selected = (years == year) & (amounts != 0)
            amounts[selected] *= struct.unpack("<d", bytes.fromhex(bits))[0]
    return amounts


def zero_origin_classes(age, receipt_code, nominal):
    """Derive source encoding classes; these do not describe respondent intent."""
    result = np.zeros(len(nominal), dtype="uint8")
    zero = nominal == 0
    result[zero & (age < 15)] = 1
    result[zero & (age >= 15) & (receipt_code == 1)] = 2
    result[zero & (age >= 15) & (receipt_code == 2)] = 3
    result[zero & (age >= 15) & (receipt_code == 0)] = 4
    return result


@dataclass(frozen=True)
class AuthenticatedIncomeObservations:
    """Immutable source authority; only the complete reconstructed join issues it."""

    _header: bytes
    _body: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "INCOME_CONSTRUCTOR_UNAVAILABLE")
        self.validate()

    @property
    def receipt(self):
        return _parse(self._header)

    @property
    def content_sha256(self):
        return _sha(self._header + self._body)

    def validate(self):
        _require(
            type(self._header) is bytes and 0 < len(self._header) <= _HEADER_MAX,
            "INCOME_HEADER_SIZE",
        )
        header = self.receipt
        _require(
            set(header) == _HEADER_KEYS and _json(header) == self._header,
            "INCOME_HEADER_SCHEMA",
        )
        rows = header["rows"]
        _require(type(rows) is int and 0 < rows <= _MAX_PERSONS, "INCOME_ROWS")
        _require(
            header["schema_version"] == 1
            and header["artifact_kind"] == ARTIFACT_KIND
            and type(self._body) is bytes
            and len(self._body) == rows * len(COLUMNS) * 8
            and _sha(self._body) == header["body_sha256"]
            and header["columns"] == list(COLUMNS)
            and header["column_dtypes"] == COLUMN_DTYPES
            and _json(header["implementation"]) == _json(_implementation())
            and header["column_sha256"]
            == {
                name: _sha(self._body[i * rows * 8 : (i + 1) * rows * 8])
                for i, name in enumerate(COLUMNS)
            },
            "INCOME_CONTENT_CHANGED",
        )

    def array(self, name):
        self.validate()
        _require(name in COLUMNS, "INCOME_COLUMN")
        n = self.receipt["rows"]
        return np.frombuffer(
            self._body,
            dtype="<f8" if name == COLUMNS[-1] else "<i8",
            count=n,
            offset=COLUMNS.index(name) * n * 8,
        )


def encode_income_observations(value):
    _require(type(value) is AuthenticatedIncomeObservations, "INCOME_AUTHORITY")
    value.validate()
    payload = (
        MAGIC + struct.pack("<I", len(value._header)) + value._header + value._body
    )
    return payload + hashlib.sha256(payload).digest()


def _reconstruct(source, ready, member_paths: Mapping[int, str | Path]):
    restoration._paths(member_paths)
    _parent(source, ready, reconstruct_money=True)
    before = _implementation()
    contract = _contract()
    _require(
        contract["price"]["resource_sha256"] == RESOURCE_PINS[1]
        and contract["price"]["target_year"] == 2024
        and price_factors(contract)
        == tuple(tuple(f) for f in ready.bindings.spec.factors),
        "INCOME_PRICE_BASIS",
    )
    person = source.frame.person
    n = len(person)
    _require(0 < n <= _MAX_PERSONS, "INCOME_ROWS")
    _require(set(COORDINATES[2:]) <= set(person), "INCOME_PARENT_ROSTER")
    _require(
        "PAW_VAL" in person
        and pd.api.types.is_numeric_dtype(person.PAW_VAL.dtype)
        and not pd.api.types.is_bool_dtype(person.PAW_VAL.dtype),
        "INCOME_INCUMBENT_DTYPE",
        "PAW_VAL",
    )
    years = np.asarray(source.scope.person_years, dtype=np.int64)
    keys = np.asarray(source.scope.person_native_keys)
    output = {
        "person_id": np.asarray(source.scope.person_ids, dtype=np.int64),
        "income_year": years,
    }
    for name in COORDINATES[2:]:
        _require(
            person[name].dtype == np.dtype("int64"), "INCOME_COORDINATE_DTYPE", name
        )
        output[name] = person[name].to_numpy(copy=True)
    output.update({name: np.empty(n, dtype=np.int64) for name in ("PAW_YN", "PAW_VAL")})
    joins = []
    with tempfile.TemporaryDirectory(prefix="microcosm-income-members-") as directory:
        for year, member, archive_pin, pin, rows, size in _MEMBER_PINS:
            staged = Path(directory) / member
            _require(
                _capture(member_paths[year], staged, size=size) == pin,
                "INCOME_SOURCE_BYTES",
            )
            raw = _read_member(staged, rows=rows)
            staged.unlink()
            positions = np.flatnonzero(years == year)
            _require(len(positions) == rows, "INCOME_COHORT_ROWS")
            recipient = pd.Index(keys[positions])
            _require(not recipient.has_duplicates, "INCOME_PERSON_KEY")
            indices = pd.Index(raw.PERIDNUM).get_indexer(recipient)
            _require(
                bool((indices >= 0).all()) and len(np.unique(indices)) == rows,
                "INCOME_KEY_COVERAGE",
            )
            joined = raw.iloc[indices]
            for name, raw_name in (
                ("source_household_id", "PH_SEQ"),
                ("A_LINENO", "A_LINENO"),
                ("A_AGE", "A_AGE"),
            ):
                _require(
                    np.array_equal(
                        output[name][positions], joined[raw_name].to_numpy()
                    ),
                    "INCOME_NATIVE_KEY_OR_AGE",
                )
            incumbent = person.PAW_VAL.iloc[positions]
            known = ~incumbent.isna().to_numpy()
            observed = joined.PAW_VAL.to_numpy(dtype=np.int64)
            _require(
                np.array_equal(
                    incumbent.to_numpy(dtype=np.float64, na_value=np.nan)[known],
                    observed[known],
                ),
                "INCOME_INCUMBENT_CONFLICT",
                "PAW_VAL",
            )
            for name in ("PAW_YN", "PAW_VAL"):
                output[name][positions] = joined[name].to_numpy(dtype=np.int64)
            classes = zero_origin_classes(
                output["A_AGE"][positions], output["PAW_YN"][positions], observed
            )
            counts = {
                name: int((classes == i).sum()) for i, name in enumerate(ZERO_CLASSES)
            }
            joins.append(
                {
                    "income_year": year,
                    "survey_year": year + 1,
                    "member": member,
                    "archive_sha256": archive_pin,
                    "member_sha256": pin,
                    "source_rows": rows,
                    "joined_rows": len(positions),
                    "unreferenced_source_rows": 0,
                    "incumbent_compared_rows": {"PAW_VAL": int(known.sum())},
                    "incumbent_conflicts": 0,
                    "native_key_or_age_conflicts": 0,
                    "zero_origin_counts": counts,
                    "paw_yn_zero_at_age_15_plus": counts["niu_in_age_universe"],
                }
            )
    output["PAW_VAL_2024_price"] = _restate_amounts(
        output["PAW_VAL"], years, ready.bindings.spec.factors
    )
    _parent(source, ready)
    _require(_json(_implementation()) == _json(before), "INCOME_IMPLEMENTATION_CHANGED")
    buffers = [
        output[name]
        .astype("<f8" if name == COLUMNS[-1] else "<i8", copy=False)
        .tobytes()
        for name in COLUMNS
    ]
    body = b"".join(buffers)
    evidence = _parse(source.source.identity)
    header = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "rows": n,
        "columns": COLUMNS,
        "column_dtypes": COLUMN_DTYPES,
        "source_identity": source.source.identity.decode(),
        "t1_sha256": evidence["person_income_attachment_sha256"],
        "source_frame_sha256": evidence["frame_sha256"],
        "scope_sha256": _parse(ready.header)["scope_sha256"],
        "native_person_keys_sha256": _sha(_json(source.scope.person_native_keys)),
        "money_header_sha256": _sha(ready.header),
        "money_content_sha256": current_money_content_sha256(ready),
        "sources": joins,
        "join_keys": ["source_year", "PERIDNUM"],
        "crosschecks": ["source_household_id", "A_LINENO", "A_AGE"],
        "price": {
            k: contract["price"][k] for k in ("algorithm", "target_year", "cells")
        }
        | {"factors": ready.bindings.spec.factors},
        # factors are the money spec's own, crosschecked above against
        # price_factors(contract); the resource never restates them itself.
        "monetary_basis": contract["monetary_basis"],
        "zero_origin_derivation": {
            "rule": "PAW_VAL_A_AGE_PAW_YN_source_encoding_v1",
            "classes": ZERO_CLASSES,
            "cohort_counts": {
                str(j["income_year"]): j["zero_origin_counts"] for j in joins
            },
        },
        "reference_period": "interview_household_one_year_after_income_year",
        "release_eligible": False,
        "implementation": before,
        "body_sha256": _sha(body),
        "column_sha256": dict(zip(COLUMNS, map(_sha, buffers), strict=True)),
        "encoding_contract": "independently_reconstructed_canonical_numeric_bytes_v1",
    }
    return AuthenticatedIncomeObservations(_json(header), body, _token=_TOKEN)


def load_authenticated_income_observations(
    source, ready, *, member_paths, candidate_path=None
):
    """Issue full PAW authority; a candidate can only match independently rebuilt bytes."""
    try:
        expected = _reconstruct(source, ready, member_paths)
        if candidate_path is not None:
            payload = encode_income_observations(expected)
            with tempfile.TemporaryDirectory(
                prefix="microcosm-income-candidate-"
            ) as directory:
                digest = _capture(
                    candidate_path, Path(directory) / FILENAME, size=len(payload)
                )
            _require(digest == _sha(payload), "INCOME_CANONICAL_BYTES")
        _parent(source, ready)
        expected.validate()
        return expected
    except MoneyRefusalError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        csv.Error,
    ):
        raise MoneyRefusalError("INCOME_SOURCE_CONTRACT") from None


def write_income_observations(source, ready, *, member_paths, output_dir):
    """Write a new immutable bundle, preserving all parent files and identities."""
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    value = load_authenticated_income_observations(
        source, ready, member_paths=member_paths
    )
    payload = encode_income_observations(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".asec-income-", dir=destination.parent
    ) as directory:
        staging = Path(directory)
        (staging / FILENAME).write_bytes(payload)
        receipt = value.receipt | {
            "content_sha256": value.content_sha256,
            "output_file": FILENAME,
            "output_sha256": _sha(payload),
        }
        (staging / "receipt.json").write_bytes(_json(receipt) + b"\n")
        destination.mkdir()
        for path in staging.iterdir():
            os.link(path, destination / path.name)
    return receipt
