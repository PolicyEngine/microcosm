"""Authenticate complete catalogues, select once, and stack original anchors.

The selection is a development declaration. Source issuers retain their own
exact-key/unknown-inclusion receipts; this successor binds the separately
demonstrated catalogue sampling relation. No allocation, calibration or donor
operation runs here. Source catalogues/parents still have full-source costs.
"""

from __future__ import annotations

import _csv
import csv
import hashlib
import json
import math
import os
import stat
import sys
import weakref
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, fields
from enum import Enum
from fractions import Fraction
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.build import survey_domain_sample
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import population as graph_population
from microcosm.graph.population import dtype_for_token, storage_equal

from . import acs_native_coverage_binding as acs_native
from . import acs_population_catalogue as acs_catalogue
from . import asec_2024_native_population as asec_native
from . import asec_current_money as current_money
from . import asec_population_catalogue as asec_catalogue
from . import graph_context, graph_sources, spine_assembly
from . import survey_catalogue_selection as selection
from . import survey_observed_age as observed_age
from . import survey_population_domains as domains
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.survey-population-preparation.v2"
REQUEST_PROTOCOL = "microcosm.us.survey-population-request.v1"
SOURCE_CODEC = "us-survey-population-source-v1"
MAX_REQUEST_BYTES = 4096
MAX_PAYLOAD_BYTES = 64 * 1024**2
_MAX_SCALAR_BYTES = 1024**2
_FRAME_FAST_STRING_CHARS = 4096
_SOURCE_ROSTER = (
    "selection-request.json",
    "acs/csv_hus.zip",
    "acs/csv_pus.zip",
    "asec/parent.h5",
    "asec/household-attachment.h5",
    "asec/person-income-attachment.h5",
    "asec/pppub23.csv",
    "asec/pppub24.csv",
    "asec/pppub25.csv",
    "asec/hhpub25.csv",
)
_TOKEN = object()
_ISSUED = {}


class SurveyPopulationPreparationError(ValueError):
    """Static refusal; messages contain no source observations or paths."""


def _require(condition, code):
    if not condition:
        raise SurveyPopulationPreparationError(code)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _check_scalars(value, depth=0):
    _require(depth <= 64, "VALUE_DEPTH")
    if type(value) is str:
        _require(len(value) <= _MAX_SCALAR_BYTES, "SCALAR_LIMIT")
    elif type(value) in (tuple, list):
        for item in value:
            _check_scalars(item, depth + 1)
    elif type(value) is dict:
        for key, item in value.items():
            _check_scalars(key, depth + 1)
            _check_scalars(item, depth + 1)


def _chunks(value):
    # A single primitive is bounded before JSON quoting. The encoder never
    # builds an unbounded complete JSON string before enforcing the transport.
    _check_scalars(value)
    yield from json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).iterencode(value)


def _encode(value, maximum=MAX_PAYLOAD_BYTES):
    result = bytearray()
    for chunk in _chunks(value):
        encoded = chunk.encode("utf-8")
        _require(len(result) + len(encoded) <= maximum, "PAYLOAD_LIMIT")
        result.extend(encoded)
    return bytes(result)


def _digest(value):
    digest = hashlib.sha256()
    for chunk in _chunks(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _catalogue_fast_path(value):
    """Recognize immutable, bounded raw records without changing admission."""
    if type(value) is not tuple or len(value) != 2:
        return False
    for group in value:
        if type(group) is not tuple:
            return False
        for record in group:
            if type(record) is not tuple or len(record) != 7:
                return False
            people = record[6]
            if type(people) is not tuple or len(people) > 20:
                return False
            if any(type(person) is not tuple or len(person) != 9 for person in people):
                return False
            characters = 0
            for row in (record[:6], *people):
                for item in row:
                    if type(item) is str:
                        if not item.isascii():
                            return False
                        characters += len(item)
                        if characters > 65_536:
                            return False
                    elif type(item) is int:
                        if not -(2**63) <= item < 2**63:
                            return False
                    else:
                        return False
    return True


def _catalogue_chunks(value):
    # Decide for the whole value before encoding. Unexpected shapes retain
    # the generic encoder's full-depth scalar pass and error precedence.
    if not _catalogue_fast_path(value):
        yield from _chunks(value)
        return
    _check_scalars(value)
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    yield "["
    for group_index, group in enumerate(value):
        if group_index:
            yield ","
        yield "["
        for record_index, record in enumerate(group):
            if record_index:
                yield ","
            # encode uses the ordinary one-shot C provider when available.
            # At most 186 leaves and 65,536 ASCII characters bound this
            # individual JSON record below 400 KiB, including escaping.
            yield encoder.encode(record)
        yield "]"
    yield "]"


def _catalogue_digest(value):
    digest = hashlib.sha256()
    for chunk in _catalogue_chunks(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _catalogue_memo(value, expected_digest):
    """Retain one proved immutable tree, never an unchecked caller digest.

    The exact-tuple/int/ASCII-str eligibility test also excludes mutable leaves
    and subclasses. Strong references prevent identity reuse. Rehash once while
    creating the memo to bind those exact immutable roots to the earlier seal.
    A replacement tree or an ineligible shape keeps the original digest path.
    """
    if not _catalogue_fast_path(value):
        return None
    if _catalogue_digest(value) != expected_digest:
        return None
    return (value[0], value[1], expected_digest)


def _memoized_catalogue_digest(value, memo=None, *, expected_digest=None):
    """Reuse only the issued preparation's unchanged immutable record roots.

    This private memo is not source authority. The owning preparation still
    performs every source/producer borrow and full physical Frame seal. Its
    original nested digest is checked independently of the memo's digest.
    """
    if (
        type(value) is tuple
        and len(value) == 2
        and type(memo) is tuple
        and len(memo) == 3
        and value[0] is memo[0]
        and value[1] is memo[1]
        and type(memo[2]) is str
        and memo[2] == expected_digest
    ):
        return memo[2]
    return _catalogue_digest(value)


def _value(value):
    if isinstance(value, Enum):
        return value.value
    if type(value) is Fraction:
        return [value.numerator, value.denominator]
    if hasattr(type(value), "__dataclass_fields__"):
        return {
            field.name: _value(getattr(value, field.name)) for field in fields(value)
        }
    if type(value) in (tuple, list):
        return [_value(item) for item in value]
    if type(value) is dict:
        return {key: _value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    _require(value is None or type(value) in (str, int, bool, float), "VALUE_TYPE")
    return value


def _plan_document(plan):
    _require(type(plan) is selection.CatalogueSelectionPlan, "SELECTION_TYPE")
    result = {
        "fraction": _value(plan.fraction),
        "seed": plan.seed,
        "supplied_households": plan.supplied_households,
    }
    budget = [0]
    for name in ("selected", "excluded", "cells"):
        result[name] = []
        for row in getattr(plan, name):
            _bounded_append(result[name], _value(row), budget)
    return result


def _root(value, *, allow_missing=False):
    _require(isinstance(value, (str, Path)), "PATH_TYPE")
    path = Path(value).absolute()
    for component in (path, *path.parents):
        if allow_missing:
            try:
                component.lstat()
            except FileNotFoundError:
                continue
        _require(not stat.S_ISLNK(component.lstat().st_mode), "SOURCE_SYMLINK")
    return path


def _stat_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _regular_reader(path, maximum):
    """Never follow a replaced final symlink or block on a substituted FIFO."""
    _root(path.parent)
    before = path.lstat()
    _require(
        stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum,
        "SOURCE_REGULAR_FILE",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and _stat_identity(opened) == _stat_identity(before),
            "SOURCE_DESCRIPTOR_CHANGED",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            yield stream, opened
            after = os.fstat(descriptor)
            current = path.lstat()
            _require(
                stat.S_ISREG(current.st_mode)
                and _stat_identity(opened)
                == _stat_identity(after)
                == _stat_identity(current),
                "SOURCE_CHANGED",
            )
    finally:
        os.close(descriptor)


def _request(source_dir):
    root = _root(source_dir)
    path = root / "selection-request.json"
    info = path.lstat()
    _require(
        stat.S_ISREG(info.st_mode) and info.st_size <= MAX_REQUEST_BYTES, "REQUEST_FILE"
    )
    with _regular_reader(path, MAX_REQUEST_BYTES) as (stream, _opened):
        payload = stream.read(MAX_REQUEST_BYTES + 1)
    _require(len(payload) <= MAX_REQUEST_BYTES, "REQUEST_LIMIT")
    document = json.loads(payload)
    _require(
        type(document) is dict
        and set(document) == {"protocol", "declaration", "fraction", "seed"},
        "REQUEST_FIELDS",
    )
    _require(
        document["protocol"] == REQUEST_PROTOCOL
        and document["declaration"] == domains.DECLARATION,
        "REQUEST_DECLARATION",
    )
    pair = document["fraction"]
    _require(
        type(pair) is list
        and len(pair) == 2
        and all(type(v) is int for v in pair)
        and pair[1] > 0,
        "REQUEST_FRACTION",
    )
    fraction = Fraction(*pair)
    _require(
        0 < fraction <= 1 and [fraction.numerator, fraction.denominator] == pair,
        "REQUEST_FRACTION",
    )
    seed = document["seed"]
    _require(type(seed) is int and 0 <= seed < 2**64, "REQUEST_SEED")
    _require(_encode(document, MAX_REQUEST_BYTES) == payload, "REQUEST_CANONICAL")
    return root, document, payload, fraction, seed


def read_survey_population_request(source_dir):
    """Decode bounded request arguments, without granting source authority."""
    try:
        _path, _document, _payload, fraction, seed = _request(source_dir)
        return fraction, seed
    except SurveyPopulationPreparationError:
        raise
    except Exception:
        raise SurveyPopulationPreparationError("REQUEST_REFUSED") from None


def _file_limits():
    # Limits bound only transport. Actual closed owners independently verify
    # their pinned source bytes and stricter member/row/receipt contracts.
    return {
        **{name: asec_native._MAX_CHECKPOINT_BYTES for name in _SOURCE_ROSTER},
        **{
            f"asec/pppub{year}.csv": asec_native.coverage_owner.literal.MAX_MEMBER_BYTES
            for year in (23, 24, 25)
        },
        "asec/hhpub25.csv": asec_native.anchor_owner._MAX_MEMBER_BYTES,
        "selection-request.json": MAX_REQUEST_BYTES,
        "acs/csv_hus.zip": acs_catalogue._ARCHIVE_BYTES,
        "acs/csv_pus.zip": acs_catalogue._ARCHIVE_BYTES,
    }


def _source_files(root):
    _root(root)
    _require(stat.S_ISDIR(root.lstat().st_mode), "SOURCE_ROOT")
    found = []
    for directory, expected in (
        (root, {"acs", "asec", "selection-request.json"}),
        (root / "acs", {"csv_hus.zip", "csv_pus.zip"}),
        (
            root / "asec",
            {Path(n).name for n in _SOURCE_ROSTER if n.startswith("asec/")},
        ),
    ):
        _require(stat.S_ISDIR(directory.lstat().st_mode), "SOURCE_DIRECTORY")
        children = tuple(directory.iterdir())
        _require(
            {p.name for p in children} == expected and len(children) == len(expected),
            "SOURCE_ROSTER",
        )
        for path in children:
            mode = path.lstat().st_mode
            _require(not stat.S_ISLNK(mode), "SOURCE_SYMLINK")
            if path.name in {"acs", "asec"} and path.parent == root:
                _require(stat.S_ISDIR(mode), "SOURCE_DIRECTORY")
            else:
                _require(stat.S_ISREG(mode), "SOURCE_REGULAR_FILE")
                found.append(path.relative_to(root).as_posix())
    _require(set(found) == set(_SOURCE_ROSTER), "SOURCE_ROSTER")
    limits = _file_limits()
    _require(
        sum(
            (root / name).lstat().st_size
            for name in ("acs/csv_hus.zip", "acs/csv_pus.zip")
        )
        <= acs_catalogue._ARCHIVE_BYTES,
        "ARCHIVE_LIMIT",
    )
    result = []
    for name in _SOURCE_ROSTER:
        path = root / name
        digest, size = hashlib.sha256(), 0
        with _regular_reader(path, limits[name]) as (stream, opened):
            while block := stream.read(1024**2):
                size += len(block)
                _require(size <= limits[name], "SOURCE_FILE_LIMIT")
                digest.update(block)
        _require(size == opened.st_size, "SOURCE_CHANGED")
        result.append([name, size, digest.hexdigest()])
    return tuple(tuple(row) for row in result)


def _file_stats(root):
    _root(root)
    result = []
    # Directory ctime/mtime also bind the exact admitted roster across the last
    # producer I/O, including a newly added unlisted entry in either source arm.
    for directory in (root, root / "acs", root / "asec"):
        info = directory.lstat()
        _require(stat.S_ISDIR(info.st_mode), "SOURCE_DIRECTORY")
        result.append(_stat_identity(info))
    for name in _SOURCE_ROSTER:
        path = root / name
        _root(path.parent)
        info = path.lstat()
        _require(stat.S_ISREG(info.st_mode), "SOURCE_REGULAR_FILE")
        result.append(_stat_identity(info))
    return tuple(result)


def _modules():
    modules = (
        sys.modules[__name__],
        domains,
        selection,
        survey_domain_sample,
        spine_assembly,
        graph_sources,
        graph_context,
        graph_population,
        observed_age,
        acs_catalogue,
        asec_catalogue,
        acs_native,
        acs_native.housing,
        acs_native.coverage,
        acs_native.coverage.literal,
        sys.modules[acs_native.housing.build_acs_pums_unit_frame.__module__],
        sys.modules[acs_native.housing.map_acs_native_inputs.__module__],
        *asec_native._modules(),
    )
    return tuple({module.__name__: module for module in modules}.values())


def _runtime_marker(value, depth=0):
    """Detach simple callable configuration, retaining opaque dependency identity."""
    _require(depth <= 16, "PRODUCER_CONFIGURATION_DEPTH")
    if type(value) in (type(None), bool, int, float, str, bytes):
        return type(value), value
    if type(value) in (tuple, list):
        return type(value), tuple(_runtime_marker(v, depth + 1) for v in value)
    if type(value) is dict:
        return tuple(
            (_runtime_marker(k, depth + 1), _runtime_marker(v, depth + 1))
            for k, v in value.items()
        )
    if isinstance(value, FunctionType):
        return value, value.__code__
    return type(value), id(value)


def _function_seal(function, depth=0):
    # contextmanager keeps executable code in both __wrapped__ and a closure;
    # the public wrapper's __code__ alone cannot bind that implementation.
    _require(depth <= 16, "PRODUCER_WRAPPER_DEPTH")
    wrapped = getattr(function, "__wrapped__", None)
    return (
        function,
        function.__code__,
        _runtime_marker(function.__defaults__),
        _runtime_marker(function.__kwdefaults__),
        tuple(_runtime_marker(c.cell_contents) for c in function.__closure__ or ()),
        _function_seal(wrapped, depth + 1)
        if isinstance(wrapped, FunctionType)
        else _runtime_marker(wrapped),
    )


def _live():
    result = {}
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[(module.__name__, name)] = _function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[(module.__name__, name)] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[(module.__name__, name, method)] = _function_seal(
                            function
                        )
    result["rng"] = (np.random.Generator, np.random.PCG64, np.random.SeedSequence)
    result["csv"] = (csv.reader, _csv.reader)
    # The bounded catalogue path uses JSONEncoder.encode and its C provider;
    # frame cells also use the same encode_basestring provider as _chunks.
    # Bind their actual live identities through this existing seal.
    result["json_catalogue"] = (
        json.JSONEncoder,
        json.encoder,
        tuple(
            _function_seal(function)
            if isinstance(function, FunctionType)
            else _runtime_marker(function)
            for function in (
                getattr(json.JSONEncoder, name, None)
                for name in ("__init__", "encode", "iterencode")
            )
        ),
        getattr(json.encoder, "c_make_encoder", None),
        getattr(json.encoder, "encode_basestring", None),
    )
    result["contract"] = (
        PROTOCOL,
        REQUEST_PROTOCOL,
        SOURCE_CODEC,
        MAX_REQUEST_BYTES,
        MAX_PAYLOAD_BYTES,
        _MAX_SCALAR_BYTES,
        _FRAME_FAST_STRING_CHARS,
        _SOURCE_ROSTER,
        domains.DECLARATION,
        domains.MAX_MEMBERS,
        domains.MAX_HOUSEHOLDS,
        domains.MAX_TOTAL_MEMBERS,
        selection._BATCH_HOUSEHOLDS,
        selection._BATCH_PEOPLE,
        observed_age.RULE,
        observed_age.AGE_CONVENTION,
        observed_age.MAX_ROWS,
        observed_age.MAX_EXACT_FLOAT64_INTEGER,
    )
    return result


def _code_bytes():
    return {
        module.__name__: _sha(Path(module.__file__).read_bytes())
        for module in _modules()
    }


def _producer():
    _require(_live() == _LIVE and _code_bytes() == _BYTES, "PRODUCER_CHANGED")
    return {
        "implementation": _BYTES,
        "acs": acs_catalogue._producer(),
        "asec": _sha(_encode(asec_catalogue._implementation())),
        "numpy": np.__version__,
        "rng": "numpy.random.PCG64/SeedSequence",
        "declaration": domains.DECLARATION,
    }


def _authority():
    return _encode(
        {
            "asec": asec_catalogue._source_authority().hex(),
            "acs": acs_native.housing._ARCHIVE_PINS,
        }
    )


def _cell(value):
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and math.isnan(value))
    ):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    _require(type(value) in (str, int, bool, float), "FRAME_CELL_TYPE")
    if type(value) is float:
        _require(math.isfinite(value), "FRAME_NONFINITE")
        return ["float", value.hex()]
    return value


def _frame_cell_encode(value, maximum=MAX_PAYLOAD_BYTES):
    """Encode one normalized cell with the exact _encode preimage and limits.

    _cell admits only None, exact bool/int/str, or its own freshly constructed
    ["float", finite_float.hex()] list. Keep normalization and the scalar walk
    before encoding; large strings/integers retain the original slow path.
    No Frame, Series, index, or normalized value is retained between calls.
    """
    value = _cell(value)
    kind = type(value)
    if not (
        value is None
        or kind is bool
        or (kind is int and -(2**63) <= value < 2**64)
        or (kind is str and len(value) <= _FRAME_FAST_STRING_CHARS)
        or kind is list
    ):
        return _encode(value, maximum)

    _check_scalars(value)
    if value is None:
        encoded = b"null"
    elif kind is bool:
        encoded = b"true" if value else b"false"
    elif kind is int:
        # Match JSON's exact-int formatter, including its decimal digit policy.
        encoded = int.__repr__(value).encode("ascii")
    elif kind is str:
        # _chunks selects this very provider with ensure_ascii=False. UTF-8
        # conversion still precedes the payload check, including surrogates.
        # At most 6 * 4096 + 2 encoded bytes, even for all control characters.
        encoded = json.encoder.encode_basestring(value).encode("utf-8")
    else:
        # Only _cell can construct this list: both strings are ASCII without
        # JSON escapes. Keep hex spelling (especially -0.0), never decimalize.
        encoded = b'["float","' + value[1].encode("ascii") + b'"]'
    _require(len(encoded) <= maximum, "PAYLOAD_LIMIT")
    return encoded


def _frame_identity(frame):
    _require(
        isinstance(frame, Frame) and frame.schema == US_SCHEMA and not frame.links,
        "FRAME_TYPE",
    )
    digest = hashlib.sha256()

    def update(value):
        digest.update(_encode(value))
        digest.update(b"\n")

    def update_cell(value):
        digest.update(_frame_cell_encode(value))
        digest.update(b"\n")

    update([list(frame.entities), list(frame.weighted_entities)])
    for entity in frame.entities:
        table = frame.table(entity)
        _require(
            type(table) is pd.DataFrame and not table.columns.has_duplicates,
            "FRAME_TABLE",
        )
        update(
            [
                entity,
                list(table.columns),
                type(table.columns).__name__,
                str(table.columns.dtype),
                list(table.columns.names),
                type(table.index).__name__,
                str(table.index.dtype),
                list(table.index.names),
            ]
        )
        for value in table.index:
            update_cell(value)
        for column in table:
            series = table[column]
            dtype = series.dtype
            update(
                [
                    column,
                    str(dtype),
                    getattr(dtype, "storage", None),
                    str(getattr(dtype, "na_value", "")),
                ]
            )
            for value in series:
                update_cell(value)
    update(
        [
            str(frame.strata.dtype),
            type(frame.strata.index).__name__,
            str(frame.strata.index.dtype),
            list(frame.strata.index.names),
            getattr(frame.strata.dtype, "storage", None),
            frame.strata.name,
        ]
    )
    for value in frame.strata.index:
        update_cell(value)
    for value in frame.strata:
        update_cell(value)
    for entity in frame.weighted_entities:
        weights = frame.weights_for(entity)
        update(
            [
                entity,
                weights.kind.value,
                str(weights.values.dtype),
                list(weights.values.shape),
            ]
        )
        digest.update(weights.values.tobytes())
    digest.update(graph_context.encode_us_frame_context(frame))
    return digest.hexdigest()


def _copy_source(frame):
    result = Frame(
        {e: frame.table(e).copy(deep=True) for e in frame.entities},
        frame.schema,
        {
            e: Weights(frame.weights_for(e).values.copy(), frame.weights_for(e).kind)
            for e in frame.weighted_entities
        },
        frame.strata.copy(deep=True),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    # Only physical string storage changes on owned copies, before the stack's
    # exact shared-dtype check. Actual values and missing masks stay identical.
    for entity in result.entities:
        for column in result.table(entity):
            original = result.table(entity)[column]
            if isinstance(original.dtype, pd.StringDtype):
                changed = original.astype(dtype_for_token("string"))
                _same_values(original, changed)
                result.table(entity)[column] = changed
    return result


def _same_values(left, right):
    _require(len(left) == len(right), "ROW_COUNT")
    _require(list(map(_cell, left)) == list(map(_cell, right)), "SOURCE_VALUE_CHANGED")


def _normalized_source_copy(frame: Frame) -> Frame:
    result = _copy_source(frame)
    _require("A_AGE" in frame.person, "OBSERVED_AGE_REQUIRED")
    result.person["age"] = observed_age.normalize_observed_age(
        frame.person["A_AGE"], frame.person.get("age")
    )
    _verify_normalized_copy(frame, result)
    return result


def _verify_normalized_copy(original: Frame, normalized: Frame):
    """Only the recorded age alias and existing string storage may differ."""
    _require(
        original.schema == normalized.schema
        and original.entities == normalized.entities
        and original.links == normalized.links == ()
        and original.metadata == normalized.metadata
        and original.mass_log == normalized.mass_log
        and original.weighted_entities == normalized.weighted_entities,
        "NORMALIZED_FRAME_CONTEXT",
    )
    for entity in original.entities:
        before, after = original.table(entity), normalized.table(entity)
        expected_columns = list(before)
        original_axis = after.columns
        if entity == "person" and "age" not in before:
            expected_columns.append("age")
            original_axis = after.columns[:-1]
        _require(
            list(after) == expected_columns
            and before.columns.identical(original_axis)
            and before.index.identical(after.index),
            "NORMALIZED_ORDERED_COLUMNS",
        )
        for column in before:
            if entity == "person" and column == "age":
                continue  # Independently reconstructed from raw observations below.
            left, right = before[column], after[column]
            if isinstance(left.dtype, pd.StringDtype):
                _require(
                    right.dtype == dtype_for_token("string"),
                    "NORMALIZED_STRING_DTYPE",
                )
                _same_values(left, right)
            else:
                _require(storage_equal(left, right), "NORMALIZED_SOURCE_STORAGE")
    _require(
        original.strata.index.identical(normalized.strata.index)
        and original.strata.name == normalized.strata.name
        and storage_equal(original.strata, normalized.strata),
        "NORMALIZED_STRATA",
    )
    for entity in original.weighted_entities:
        left, right = original.weights_for(entity), normalized.weights_for(entity)
        _require(
            left.kind is right.kind
            and left.values.dtype == right.values.dtype
            and left.values.tobytes() == right.values.tobytes(),
            "NORMALIZED_WEIGHTS",
        )
    expected = observed_age.normalize_observed_age(
        original.person["A_AGE"], original.person.get("age")
    )
    _require(
        storage_equal(expected, normalized.person["age"]), "NORMALIZED_AGE_IDENTITY"
    )


def _bounded_append(rows, row, budget):
    size = len(_encode(row)) + 1
    _require(budget[0] + size <= MAX_PAYLOAD_BYTES, "ORIGIN_LIMIT")
    budget[0] += size
    rows.append(row)


def _origins(frame, sources, selected, native_receipts):
    selected_by_key = {(r.key.source.value, r.key.native_id): r for r in selected}
    native_hh, native_people = {}, {}
    for row in native_receipts["asec"]["households"]:
        native_hh[("asec", row["household_id"])] = row["source"]["H_SEQ"]
    asec_rows = {row["person_id"]: row for row in native_receipts["asec"]["persons"]}
    for channel, source in sources.items():
        for row in source.table("household").itertuples(index=False):
            if channel == "acs":
                native_hh[(channel, int(row.household_id))] = row.SERIALNO
        for row in source.person.itertuples(index=False):
            if channel == "acs":
                person_key, line = str(row.SPORDER), str(row.SPORDER)
            else:
                original = asec_rows[int(row.person_id)]
                person_key, line = original["PERIDNUM"], str(original["A_LINENO"])
            native_people[(channel, int(row.person_id))] = (
                person_key,
                line,
                int(row.person_household_id),
            )
    maps, entities, households, people, budget = {}, {}, [], [], [0]
    household_positions = {}
    for channel, source in sources.items():
        ids = source.table("household").household_id
        positions = {int(value): index for index, value in enumerate(ids)}
        _require(len(positions) == len(ids), "SOURCE_ID_COLLISION")
        household_positions[channel] = positions
    for entity in frame.entities:
        table = frame.table(entity)
        ids = US_SCHEMA.entity_id_column(entity)
        channels, previous = (
            support_channel_column(entity),
            spine_source_id_column(entity),
        )
        provenance = set(spine_assembly._support_metadata_columns(entity))
        expected_columns = provenance | {
            column for source in sources.values() for column in source.table(entity)
        }
        _require(set(table) == expected_columns, "UNDECLARED_STACK_COLUMN")
        records, mapping = [], {}
        for new_id, channel, source_id in table[[ids, channels, previous]].itertuples(
            index=False, name=None
        ):
            key = (channel, int(source_id))
            _require(key not in mapping, "SOURCE_ID_COLLISION")
            mapping[key] = int(new_id)
            _bounded_append(records, [int(new_id), channel, int(source_id)], budget)
        expected = {
            (channel, int(i))
            for channel, source in sources.items()
            for i in source.table(entity)[ids]
        }
        _require(set(mapping) == expected, "COMPLETE_ENTITY_ORIGIN")
        maps[entity], entities[entity] = mapping, records
    for entity in frame.entities:
        table = frame.table(entity)
        id_column = US_SCHEMA.entity_id_column(entity)
        for channel, source in sources.items():
            arm = table.loc[table[support_channel_column(entity)].eq(channel)]
            old_ids = arm[spine_source_id_column(entity)].to_numpy()
            original = (
                source.table(entity).set_index(id_column, drop=False).loc[old_ids]
            )
            for column in original:
                if column == id_column:
                    expected = [
                        maps[entity][(channel, int(v))] for v in original[column]
                    ]
                elif entity == "person" and column in {
                    US_SCHEMA.membership_column(e) for e in US_SCHEMA.group_entities
                }:
                    group = next(
                        e
                        for e in US_SCHEMA.group_entities
                        if US_SCHEMA.membership_column(e) == column
                    )
                    expected = [
                        maps[group][(channel, int(v))] for v in original[column]
                    ]
                else:
                    expected = original[column]
                _same_values(expected, arm[column])
            for column in (
                set(arm)
                - set(original)
                - set(spine_assembly._support_metadata_columns(entity))
            ):
                _require(arm[column].isna().all(), "ABSENT_SOURCE_VALUE_INVENTED")
            if entity == "person":
                source_positions = {
                    int(value): index
                    for index, value in enumerate(source.person.person_id)
                }
                expected_strata = source.strata.iloc[
                    [source_positions[int(value)] for value in old_ids]
                ]
                receiving_positions = np.flatnonzero(
                    table[support_channel_column(entity)].eq(channel).to_numpy()
                )
                _same_values(expected_strata, frame.strata.iloc[receiving_positions])
    household_table = frame.table("household")
    for position, (new_id, channel, source_id) in enumerate(entities["household"]):
        raw = native_hh[(channel, source_id)]
        chosen = selected_by_key.pop((channel, raw), None)
        _require(chosen is not None, "SELECTED_SOURCE_IDENTITY")
        original = chosen.original_design_weight
        source = sources[channel]
        source_position = household_positions[channel][source_id]
        _require(
            source.weights_for("household").kind is WeightKind.DESIGN
            and frame.weights_for("household").kind is WeightKind.DESIGN,
            "ORIGINAL_WEIGHT_KIND",
        )
        expected = np.array([float(original)], dtype=np.float64).tobytes()
        _require(
            source.weights_for("household")
            .values[source_position : source_position + 1]
            .tobytes()
            == expected
            == frame.weights_for("household").values[position : position + 1].tobytes(),
            "ORIGINAL_ANCHOR_CHANGED",
        )
        _bounded_append(
            households,
            {
                "household_id": new_id,
                "source": channel,
                "source_year": 2024,
                "survey_year": 2024 if channel == "acs" else 2025,
                "raw_native_id": raw,
                "selected_receiving_household_id": source_id,
                "original_anchor": _value(original),
            },
            budget,
        )
    _require(
        not selected_by_key and len(households) == len(household_table),
        "SELECTED_HOUSEHOLD_ROSTER",
    )
    for new_id, channel, source_id in entities["person"]:
        person_key, line, household_id = native_people[(channel, source_id)]
        _bounded_append(
            people,
            [
                new_id,
                channel,
                2024,
                2024 if channel == "acs" else 2025,
                native_hh[(channel, household_id)],
                person_key,
                line,
                source_id,
                household_id,
                maps["household"][(channel, household_id)],
            ],
            budget,
        )
    return {
        "households": households,
        "entities": entities,
        "persons": {
            "columns": [
                "person_id",
                "source",
                "source_year",
                "survey_year",
                "raw_native_household_id",
                "raw_native_person_id",
                "native_line_numeric_original",
                "selected_receiving_person_id",
                "selected_receiving_household_id",
                "household_id",
            ],
            "rows": people,
        },
    }


@dataclass(frozen=True)
class _State:
    root: Path
    files: tuple
    file_stats: tuple
    producer: bytes
    authority: bytes
    catalogues: tuple
    native: tuple
    attached: tuple
    source_frames: tuple
    source_frame_seals: tuple
    frame: Frame
    identity: str
    context: bytes
    plan: selection.CatalogueSelectionPlan
    plan_sha256: str
    nested: tuple
    acs_catalogue_memo: tuple | None = None


def _attached(state):
    return tuple(value.payload for value in (*state.catalogues, *state.native))


def _nested_seals(
    catalogues, native, *, catalogue_memo=None, expected_catalogue_digest=None
):
    """Pure final borrowed-object seals after all source/producer file checks.

    These inspect the actual retained issued owners, never construct issuer
    authority from decoded records. The ancestor Frames still impose their
    existing full-parent verification cost.
    """
    acs = acs_native._owned(native[0])
    acs_cat = acs_catalogue._lookup(catalogues[0])
    values = [
        id(acs),
        id(acs.prepared),
        id(acs.prepared.source),
        id(acs.literal),
        acs.prepared.receipt_json,
        acs.prepared.source.receipt_json,
        acs.prepared.source.projection_json,
        acs.literal.payload,
        _encode(
            [
                [str(path) for _, path in sorted(paths.items())]
                for paths in acs.snapshots
            ]
        ),
        _memoized_catalogue_digest(
            (acs_cat.records, acs_cat.vacancies),
            catalogue_memo,
            expected_digest=expected_catalogue_digest,
        ),
    ]
    for module, value in ((asec_catalogue, catalogues[1]), (asec_native, native[1])):
        entry = module._ISSUED.get(id(value))
        _require(
            entry is not None and entry[0]() is value and entry[1] == value.payload,
            "ANCESTOR_ISSUANCE_CHANGED",
        )
        state = entry[2]
        attached = asec_native._attached_evidence(
            state.parent, state.coverage, state.anchors, state.fields
        )
        values.extend(
            (
                id(state),
                id(state.parent),
                id(attached[0]),
                *attached[1:],
                asec_native._frame_identity(state.parent.frame),
            )
        )
        if module is asec_catalogue:
            values.append(asec_catalogue._current_records_identity(state))
    return tuple(values)


def _pure_final(state):
    _require(
        _live() == _LIVE and _authority() == state.authority, "FINAL_AUTHORITY_CHANGED"
    )
    _require(_attached(state) == state.attached, "ATTACHED_EVIDENCE_CHANGED")
    _require(
        _nested_seals(
            state.catalogues,
            state.native,
            catalogue_memo=state.acs_catalogue_memo,
            expected_catalogue_digest=state.nested[9],
        )
        == state.nested,
        "NESTED_EVIDENCE_CHANGED",
    )
    _require(_digest(_plan_document(state.plan)) == state.plan_sha256, "PLAN_CHANGED")
    _require(
        _frame_identity(state.frame) == state.identity
        and graph_context.encode_us_frame_context(state.frame) == state.context,
        "PREPARED_FRAME_CHANGED",
    )
    _require(
        tuple(
            (
                acs_native._frame_sha256(frame)
                if i == 0
                else asec_native._frame_identity(frame)
            )
            for i, frame in enumerate(state.source_frames)
        )
        == state.source_frame_seals,
        "SOURCE_FRAME_CHANGED",
    )


def _validate(state):
    _pure_final(state)
    acs_catalogue.verify_acs_source_catalogue(state.catalogues[0])
    asec_catalogue.verify_asec_source_catalogue(state.catalogues[1])
    acs_native.verify_acs_native_coverage(state.native[0], state.source_frames[0])
    state.native[1].validate()
    _require(_source_files(state.root) == state.files, "SOURCE_CHANGED")
    _require(_encode(_producer()) == state.producer, "PRODUCER_CHANGED")
    _require(_file_stats(state.root) == state.file_stats, "SOURCE_STAT_CHANGED")
    _pure_final(state)


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class AuthenticatedSurveyPopulationPreparation:
    payload: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "ISSUANCE_CONSTRUCTOR")

    def _checked(self):
        try:
            entry = _ISSUED.get(id(self))
            _require(
                type(self) is AuthenticatedSurveyPopulationPreparation
                and entry is not None
                and entry[0]() is self
                and type(self.payload) is bytes
                and self.payload == entry[1],
                "UNISSUED_OR_CHANGED",
            )
            _validate(entry[2])
            _require(
                _ISSUED.get(id(self)) is entry
                and type(self.payload) is bytes
                and self.payload == entry[1],
                "UNISSUED_OR_CHANGED",
            )
            return entry
        except SurveyPopulationPreparationError:
            raise
        except Exception:
            raise SurveyPopulationPreparationError(
                "PREPARATION_VERIFICATION_REFUSED"
            ) from None

    def validate(self):
        self._checked()

    def checked_view(self):
        """Borrow one checked bundle; the view itself grants no authority."""
        _reference, payload, state = self._checked()
        return CheckedSurveyPopulationView(
            payload, state.context, state.frame, state.plan, json.loads(payload)
        )

    @property
    def frame(self):
        return self._checked()[2].frame

    @property
    def context(self):
        return self._checked()[2].context

    @property
    def selection_plan(self):
        return self._checked()[2].plan

    @property
    def receipt(self):
        return json.loads(self._checked()[1])

    def to_bytes(self):
        return self._checked()[1]


@dataclass(frozen=True, slots=True)
class CheckedSurveyPopulationView:
    """A checked borrow's values, not an issued source or reusable certificate."""

    payload: bytes
    context: bytes
    frame: Frame
    selection_plan: selection.CatalogueSelectionPlan
    receipt: dict


def verify_survey_population_preparation(value):
    _require(
        type(value) is AuthenticatedSurveyPopulationPreparation, "PREPARATION_TYPE"
    )
    value._checked()
    return value


def verify_materialized_survey_population(preparation, frame):
    _require(
        type(preparation) is AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    _require(_frame_identity(frame) == entry[2].identity, "MATERIALIZED_FRAME_CHANGED")
    _validate(entry[2])
    _require(_frame_identity(frame) == entry[2].identity, "MATERIALIZED_FRAME_CHANGED")
    _require(preparation.payload == entry[1], "UNISSUED_OR_CHANGED")


def prepare_authenticated_survey_population(
    source_dir, *, snapshot_root, fraction, seed, candidate=None
):
    """Reconstruct actual source authority before accepting candidate bytes."""
    try:
        _require(type(candidate) is bytes or candidate is None, "CANDIDATE_TYPE")
        _require(
            candidate is None or len(candidate) <= MAX_PAYLOAD_BYTES, "CANDIDATE_LIMIT"
        )
        root, request, request_bytes, requested_fraction, requested_seed = _request(
            source_dir
        )
        _require(
            type(fraction) is Fraction
            and fraction == requested_fraction
            and type(seed) is int
            and seed == requested_seed,
            "REQUEST_ARGUMENTS",
        )
        snapshots = _root(snapshot_root, allow_missing=True)
        _require(
            not snapshots.is_relative_to(root) and not root.is_relative_to(snapshots),
            "SNAPSHOT_LOCATION",
        )
        snapshots.mkdir(parents=True, exist_ok=True)
        authority, files_before, producer = (
            _authority(),
            _source_files(root),
            _encode(_producer()),
        )
        _require(
            files_before[0]
            == ("selection-request.json", len(request_bytes), _sha(request_bytes)),
            "REQUEST_CHANGED",
        )
        file_stats = _file_stats(root)
        kwargs = dict(
            parent_path=root / "asec/parent.h5",
            household_attachment_path=root / "asec/household-attachment.h5",
            person_income_attachment_path=root / "asec/person-income-attachment.h5",
            person_member_paths={
                2022: root / "asec/pppub23.csv",
                2023: root / "asec/pppub24.csv",
                2024: root / "asec/pppub25.csv",
            },
            household_member_path=root / "asec/hhpub25.csv",
        )
        acs = acs_catalogue.issue_acs_source_catalogue(
            root / "acs", snapshot_root=snapshots
        )
        asec = asec_catalogue.issue_asec_source_catalogue(**kwargs)
        acs_rows, asec_rows = acs.households, asec.households
        plan = selection.plan_catalogue_selection(
            acs_households=acs_rows,
            asec_households=asec_rows,
            fraction=fraction,
            seed=seed,
        )
        acs_keys = tuple(
            r.key.native_id for r in plan.selected if r.key.source is domains.Source.ACS
        )
        asec_keys = tuple(
            (2024, int(r.key.native_id))
            for r in plan.selected
            if r.key.source is domains.Source.ASEC
        )
        _require(
            acs_keys and asec_keys and len(set(asec_keys)) == len(asec_keys),
            "SELECTED_SOURCE_SUPPORT",
        )
        actual_acs = acs_native.issue_acs_native_coverage(
            root / "acs", snapshot_root=snapshots, serialnos=acs_keys
        )
        actual_asec = asec_native.load_authenticated_asec_2024_native_population(
            **kwargs, selected_households=asec_keys
        )
        native_frames = actual_acs.frame, actual_asec.frame
        source_seals = (
            acs_native._frame_sha256(native_frames[0]),
            asec_native._frame_identity(native_frames[1]),
        )
        source_copies = {
            channel: _normalized_source_copy(frame)
            for channel, frame in zip(("acs", "asec"), native_frames, strict=True)
        }
        stacked = spine_assembly.stack_survey_spines(source_copies)
        frame = stacked.frame
        transitions = graph_sources._canonical_assembly(
            frame, tuple(source_copies.values())
        )
        context = graph_context.encode_us_frame_context(frame)
        native_receipts = {"acs": actual_acs.receipt, "asec": actual_asec.receipt}
        for channel, original in zip(("acs", "asec"), native_frames, strict=True):
            _verify_normalized_copy(original, source_copies[channel])
        origins = _origins(frame, source_copies, plan.selected, native_receipts)
        origins["storage_transitions"] = list(transitions)
        origins["observed_age_normalization"] = {
            "rule": observed_age.rule_document(),
            "sources": {
                channel: {
                    "native_frame_sha256": source_seals[i],
                    "normalized_frame_sha256": _frame_identity(source_copies[channel]),
                    "common_age_preexisting": "age" in native_frames[i].person,
                }
                for i, channel in enumerate(("acs", "asec"))
            },
        }
        documents = {"acs": acs.receipt, "asec": asec.receipt}
        catalogues = {
            "acs": {
                "receipt_sha256": _sha(acs.to_bytes()),
                "records_sha256": documents["acs"]["counts"]["canonical_record_sha256"],
                "counts": documents["acs"]["counts"],
            },
            "asec": {
                "receipt_sha256": _sha(asec.to_bytes()),
                "records_sha256": documents["asec"]["records"]["sha256"],
                "counts": documents["asec"]["counts"],
            },
        }
        identity = _frame_identity(frame)
        payload = _encode(
            {
                "protocol": PROTOCOL,
                "request": request,
                "request_sha256": _sha(request_bytes),
                "source_files": files_before,
                "producer": json.loads(producer),
                "catalogues": catalogues,
                "native": {
                    channel: {
                        "receipt_sha256": _sha(value.payload),
                        "frame_sha256": source_seals[i],
                        "households": native_frames[i].n("household"),
                        "persons": native_frames[i].n("person"),
                    }
                    for i, (channel, value) in enumerate(
                        (("acs", actual_acs), ("asec", actual_asec))
                    )
                },
                "selection": _plan_document(plan),
                "origins": origins,
                "frame_sha256": identity,
                "context_sha256": _sha(context),
                "release_eligible": False,
            }
        )
        nested = _nested_seals((acs, asec), (actual_acs, actual_asec))
        acs_owned = acs_catalogue._lookup(acs)
        catalogue_memo = _catalogue_memo(
            (acs_owned.records, acs_owned.vacancies), nested[9]
        )
        state = _State(
            root,
            files_before,
            file_stats,
            producer,
            authority,
            (acs, asec),
            (actual_acs, actual_asec),
            tuple(value.payload for value in (acs, asec, actual_acs, actual_asec)),
            native_frames,
            source_seals,
            frame,
            identity,
            context,
            plan,
            _digest(_plan_document(plan)),
            nested,
            catalogue_memo,
        )
        _validate(state)
        _require(candidate is None or candidate == payload, "CANDIDATE_MISMATCH")
        result = AuthenticatedSurveyPopulationPreparation(payload, _token=_TOKEN)
        key = id(result)

        def cleanup(reference):
            entry = _ISSUED.get(key)
            if entry is not None and entry[0] is reference:
                del _ISSUED[key]

        _ISSUED[key] = (weakref.ref(result, cleanup), payload, state)
        _pure_final(state)
        _require(
            type(result.payload) is bytes
            and result.payload == payload
            and _ISSUED[key][0]() is result,
            "UNISSUED_OR_CHANGED",
        )
        return result
    except SurveyPopulationPreparationError:
        raise
    except Exception:
        raise SurveyPopulationPreparationError("PREPARATION_ISSUANCE_REFUSED") from None


_BYTES = _code_bytes()


def _current_survey_wage_projection(preparation, entry):
    """Project one just-checked retained owner; bytes grant no source authority.

    The caller must have obtained ``entry`` from ``preparation._checked()`` in
    the current operation, and must seal its receiving Populations afterwards.
    The one ready() call retains the existing all-field/full-parent admission.
    No old reported-income contract or prior-wage column is manufactured.
    """
    _require(
        type(preparation) is AuthenticatedSurveyPopulationPreparation
        and _ISSUED.get(id(preparation)) is entry
        and entry[0]() is preparation
        and type(preparation.payload) is bytes
        and preparation.payload == entry[1],
        "WAGE_PREPARATION_ISSUANCE",
    )
    state = entry[2]
    native_entry = asec_native._ISSUED.get(id(state.native[1]))
    _require(
        native_entry is not None
        and native_entry[0]() is state.native[1]
        and state.native[1].payload == native_entry[1],
        "WAGE_NATIVE_ISSUANCE",
    )
    parent = native_entry[2].parent
    ready = parent.ready()  # Exactly once, outside both row and feature loops.
    field = ready.field("WSAL_VAL")
    domain = next(d for d in parent.spec.fields if d.name == "WSAL_VAL")
    scope = parent.scope
    positions = {pid: i for i, pid in enumerate(scope.person_ids)}
    _require(len(positions) == len(scope.person_ids), "WAGE_PARENT_ROSTER")
    native_ids = set(state.source_frames[1].person.person_id)
    rows, budget = [], [0]
    people = state.frame.person
    selected = people.loc[people[support_channel_column("person")].eq("asec")]
    _require(
        set(selected[spine_source_id_column("person")]) == native_ids,
        "WAGE_SELECTED_ROSTER",
    )
    for stacked, native in selected[
        ["person_id", spine_source_id_column("person")]
    ].itertuples(index=False, name=None):
        _require(int(native) in positions, "WAGE_PARENT_MEMBER")
        i = positions[int(native)]
        _require(scope.person_years[i] == 2024, "WAGE_CURRENT_COHORT")
        amount = field.amount_bytes[8 * i : 8 * (i + 1)]
        number = np.frombuffer(amount, dtype="<f8")[0]
        _require(
            field.validity_bytes[i] == 1 and np.isfinite(number) and number >= 0,
            "WAGE_CURRENT_FEATURE",
        )
        _bounded_append(
            rows,
            [
                int(stacked),
                int(native),
                2024,
                amount.hex(),
                field.status_bytes[i],
                field.validity_bytes[i],
                field.zero_origin_bytes[i],
            ],
            budget,
        )
    document = {
        "preparation_sha256": _sha(entry[1]),
        "asec_native_sha256": _sha(native_entry[1]),
        "money_header": json.loads(ready.header),
        "money_header_sha256": _sha(ready.header),
        "domain": {f.name: getattr(domain, f.name) for f in fields(domain)},
        "zero_origin_code": int(current_money._origin_code(parent.spec, "WSAL_VAL")),
        "columns": [
            "stacked_person_id",
            "native_person_id",
            "income_year",
            "amount_f64le_hex",
            "status",
            "validity",
            "zero_origin",
        ],
        "rows": rows,
    }
    payload = _encode(document)
    # ready() may perform I/O. Recheck the actual retained entries and all pure
    # owner seals after it; this does not reread the national catalogues.
    _pure_final(state)
    _require(
        _ISSUED.get(id(preparation)) is entry
        and preparation.payload == entry[1]
        and asec_native._ISSUED.get(id(state.native[1])) is native_entry
        and state.native[1].payload == native_entry[1]
        and native_entry[2].parent is parent,
        "WAGE_FINAL_ISSUANCE",
    )
    return payload


_LIVE = _live()
