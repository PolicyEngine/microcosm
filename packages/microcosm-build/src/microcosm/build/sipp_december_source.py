"""Authenticated, bounded December evidence from the closed 2023 SIPP parent.

No engine, fitting, recipient mapping or year transfer. See
docs/us-sipp-december-source.md for the local single-writer and ownership boundary.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from itertools import islice
from pathlib import Path
from typing import Literal

from microcosm.graph.canonical import canonical_json, sha256_domain
from microcosm.graph.kernel import source_hash
from microcosm.graph.keys import source_content_key
from microcosm.graph.store import ContentStore, StoreError

from . import sipp_measurements as m

FIELDS = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "ERESIDENCEID",
    "WPFINWGT",
    "THEQ_VEH",
    "AHEQ_VEH",
    "THVAL_VEH",
    "AHVAL_VEH",
    "THDEBT_VEH",
    "AHDEBT_VEH",
    "TVEH_NUM",
    "AVEH_NUM",
    "TVAL_BANK",
    "AVAL_BANK",
    "TAGE_EHC",
)
_COPIED = ("THEQ_VEH", "THVAL_VEH", "THDEBT_VEH", "TVEH_NUM")
type Selection = Literal["first_32_ssuids", "all_ssuids"]
_MODES = ("first_32_ssuids", "all_ssuids")
_SOURCE_NAME = "sipp_2023_pu2023"
_NAMESPACE = "sipp-december-source-v1"
_PAYLOAD_MAX = 64 * 1024 * 1024
_META_MAX = 1024 * 1024
_RECORD_MAX = 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_MIN_RAM = 8 * 1024**3
_KEY = re.compile(r"[0-9a-f]{64}\Z")
_MODULES = (
    "microcosm.build",
    "microcosm.build.sipp_december_source",
    "microcosm.build.sipp_measurements",
    "microcosm.frame",
    "microcosm.graph",
    "microcosm.graph.canonical",
    "microcosm.graph.codecs",
    "microcosm.graph.decl",
    "microcosm.graph.errors",
    "microcosm.graph.kernel",
    "microcosm.graph.keys",
    "microcosm.graph.store",
)
_DEPENDENCIES = (
    "microcosm-build",
    "microcosm-frame",
    "microcosm-graph",
    "numpy",
    "pandas",
)


class SourceError(ValueError):
    """Sanitized refusal at the December source authority boundary."""


@dataclass(frozen=True)
class _Pins:
    sha256: str
    size: int
    metadata_sha256: str
    registry_sha256: str
    revision: str
    dictionary_sha256: str
    expected_counts: tuple[int, ...] | None


# Test fixtures replace this private object; callers cannot supply pin authority.
_PINS = _Pins(
    "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2",
    3726010471,
    m.SIPP_2023_MASTER_SHA256,
    "085658b4a9c8cba0e59624a64ae998c3e07d63ceba2243ab92904a0bef5f8402",
    "21280dca5995e978d706740a8a4b9b7860cfd7b6",
    "0f1c5fbd29b24d3f1c18e7168a0f6d9fdfa7578b58d247fa9f475093a6db894f",
    (39513, 17646, 17314, 318, 29),
)


@dataclass(frozen=True)
class SippDecemberSource:
    """Owned reconstructed evidence; no training or downstream mapping approval."""

    households: tuple[m.DecemberHousehold, ...]
    bank: tuple[m.Measurement, ...]
    receipt_json: bytes
    receipt_key: str
    projection_key: str
    producer_binding: str
    run_information_key: str = field(compare=False)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(domain: str, value: object) -> str:
    return sha256_domain(domain, canonical_json(value))


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory(path: Path) -> None:
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise SourceError("NONREGULAR_DIRECTORY")


def _open_regular(path: Path, flags: int = os.O_RDONLY) -> int:
    fd = os.open(path, flags | os.O_NONBLOCK | os.O_NOFOLLOW, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SourceError("NONREGULAR_FILE")
    return fd


def _copy_regular(
    source: Path, destination: Path, cap: int, *, exact_size: int | None = None
) -> str:
    """Copy from one nonblocking descriptor; cap real reads and detect changes."""
    with os.fdopen(_open_regular(source), "rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > cap or (
            exact_size is not None and before.st_size != exact_size
        ):
            raise SourceError("SOURCE_SIZE")
        count = 0
        digest = hashlib.sha256()
        with destination.open("xb") as output:
            while chunk := stream.read(min(_CHUNK_BYTES, cap - count + 1)):
                count += len(chunk)
                if count > cap:
                    raise SourceError("SOURCE_TOO_LARGE")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if (
            count != before.st_size
            or _identity(before) != _identity(os.fstat(stream.fileno()))
            or _identity(before) != _identity(source.lstat())
        ):
            raise SourceError("SOURCE_CHANGED")
    destination.chmod(0o400)
    return digest.hexdigest()


def _available_memory() -> int:
    """Operational available-memory estimate; never process RSS."""
    if "SC_AVPHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "darwin":
        result = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, check=True, timeout=5
        )
        text = result.stdout.decode("ascii")
        page_size = re.search(r"page size of (\d+) bytes", text)
        counts = [
            re.search(rf"{name}:\s+(\d+)\.", text)
            for name in ("Pages free", "Pages inactive", "Pages speculative")
        ]
        if page_size is None or any(c is None for c in counts):
            raise SourceError("AVAILABLE_MEMORY_UNSUPPORTED")
        size = int(page_size.group(1))
        counts = [int(c.group(1)) for c in counts]
        return size * sum(counts)
    raise SourceError("AVAILABLE_MEMORY_UNSUPPORTED")


def _resources(namespace: Path, pins: _Pins) -> dict:
    failures = []
    for path in namespace.glob("snapshot-*"):
        _directory(path)
        parent = path / "parent.bin"
        if os.path.lexists(parent):
            meta = parent.lstat()
            if not stat.S_ISREG(meta.st_mode):
                raise SourceError("INVALID_RETAINED_SNAPSHOT")
            failures.append(meta.st_size)
        else:
            failures.append(0)
    free = shutil.disk_usage(namespace).free
    available = _available_memory()
    required_disk = pins.size + 4 * _PAYLOAD_MAX + 1024**3
    if free < required_disk or available < _MIN_RAM:
        raise SourceError("INSUFFICIENT_LOCAL_RESOURCES")
    return {
        "free_disk_bytes": free,
        "required_additional_disk_bytes": required_disk,
        "available_memory_estimate_bytes": available,
        "available_memory_basis": "sysconf_available_pages_or_mac_free_inactive_speculative",
        "retained_snapshot_count": len(failures),
        "retained_snapshot_bytes": sum(failures),
        "occupied_snapshot_bytes_subtracted_again": False,
    }


@contextmanager
def _locked(store: ContentStore):
    _directory(store.root)
    namespace = store.root / _NAMESPACE
    namespace.mkdir(mode=0o700, exist_ok=True)
    _directory(namespace)
    with os.fdopen(
        _open_regular(namespace / "producer.lock", os.O_RDWR | os.O_CREAT), "r+b"
    ) as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SourceError("PRODUCER_BUSY") from None
        try:
            yield namespace
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _contract(pins: _Pins) -> tuple[str, dict]:
    modules = tuple(importlib.import_module(name) for name in _MODULES)
    closure = {
        module.__name__: _sha(Path(module.__file__).read_bytes()) for module in modules
    }
    versions = {name: importlib.metadata.version(name) for name in _DEPENDENCIES}
    value = {
        "parent": {
            "name": _SOURCE_NAME,
            "sha256": pins.sha256,
            "bytes": pins.size,
            "content_key": _key("source", (_SOURCE_NAME, pins.sha256, pins.size)),
            "revision": pins.revision,
        },
        "master_sha256": pins.metadata_sha256,
        "registry_sha256": pins.registry_sha256,
        "dictionary_sha256": pins.dictionary_sha256,
        "bank_dictionary_pages": [191, 797],
        "source_years": {"survey": 2023, "reference": 2022},
        "fields": FIELDS,
        "parser": {
            "version": "sipp-december-lexical-parser/1",
            "encoding": "utf-8-strict-no-BOM",
            "delimiter": "|",
            "quoting": "QUOTE_NONE",
            "blank_physical_records": "refuse",
            "full_header_width_every_record": True,
            "record_max_bytes": _RECORD_MAX,
            "csv_field_size_limit": csv.field_size_limit(),
            "lexical_values": "strings_no_NA_conversion",
            "nondecember_keys": "not_validated",
        },
        "selection": "sipp-december-whole-ssuid-selection/1",
        "serializer": "sipp-december-explicit-json/1",
        "decimal": "finite_fixed_point_preserve_scale_sign",
        "rules": [m.MEASUREMENT_RULE_ID, "sipp_december_bank_assets_v1"],
        "copied_fields": _COPIED,
        "expected_counts": pins.expected_counts,
        "implementation": {
            "module_sha256": closure,
            "dependency_versions": versions,
            "python": sys.version,
            "source_hash": source_hash(*modules, dependencies=_DEPENDENCIES),
        },
    }
    return _key("sipp-december-producer-binding/1", value), value


def _line(stream) -> str | None:
    data = stream.readline(_RECORD_MAX + 1)
    if not data:
        return None
    if len(data) > _RECORD_MAX:
        raise SourceError("RECORD_TOO_LARGE")
    if data.startswith(b"\xef\xbb\xbf"):
        raise SourceError("UTF8_BOM")
    if data.endswith(b"\n"):
        data = data[:-1]
        if data.endswith(b"\r"):
            data = data[:-1]
    if not data or b"\r" in data or b"\x00" in data:
        raise SourceError("INVALID_PHYSICAL_RECORD")
    return data.decode("utf-8", errors="strict")


def _parts(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="|", quoting=csv.QUOTE_NONE, strict=True))


def _parse(snapshot: Path, chunksize: int) -> tuple[list[m.SippRow], dict]:
    rows = []
    months = Counter()
    with os.fdopen(_open_regular(snapshot), "rb") as stream:
        before = os.fstat(stream.fileno())
        first = _line(stream)
        if first is None:
            raise SourceError("EMPTY_PARENT")
        header = _parts(first)
        if (
            any(not f for f in header)
            or len(set(header)) != len(header)
            or not set(FIELDS) <= set(header)
        ):
            raise SourceError("INVALID_HEADER")
        positions = [header.index(f) for f in FIELDS]
        batch = []

        def consume():
            for values in batch:
                _, month = m._number(values, "MONTHCODE", 1, 12, -9)
                if month is None:
                    raise SourceError("INVALID_ROSTER_MONTH")
                months[str(int(month))] += 1
                if month == 12:
                    rows.append(m.SippRow(values))
            batch.clear()

        while (line := _line(stream)) is not None:
            tokens = _parts(line)
            if len(tokens) != len(header):
                raise SourceError("ROW_WIDTH")
            batch.append(dict(zip(FIELDS, (tokens[p] for p in positions), strict=True)))
            if len(batch) == chunksize:
                consume()
        consume()
        if _identity(before) != _identity(os.fstat(stream.fileno())) or _identity(
            before
        ) != _identity(snapshot.lstat()):
            raise SourceError("SNAPSHOT_CHANGED")
    return rows, {
        "header_columns": len(header),
        "physical_records": sum(months.values()),
        "month_records": dict(sorted(months.items())),
    }


def _selected_ssuids(
    keys: tuple[tuple[str, int], ...], selection: Selection
) -> tuple[str, ...]:
    # This boundary intentionally accepts only canonical identity tuples.
    if any(
        type(k) is not tuple
        or len(k) != 2
        or type(k[0]) is not str
        or type(k[1]) is not int
        for k in keys
    ):
        raise SourceError("INVALID_SELECTOR_KEYS")
    available = tuple(sorted({k[0] for k in keys}))
    return available[:32] if selection == "first_32_ssuids" else available


def _decimal(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if not value.is_finite():
        raise SourceError("NONFINITE_MEASUREMENT")
    return format(value, "f")


def _measurement(value: m.Measurement) -> dict:
    return {
        "name": value.name,
        "concept": value.concept,
        "source": {
            "data_sha256": value.source.data_sha256,
            "metadata_sha256": value.source.metadata_sha256,
            "survey_year": value.source.survey_year,
            "reference_year": value.source.reference_year,
        },
        "basis": value.basis,
        "unit": value.unit,
        "value": _decimal(value.value),
        "status": value.status,
        "reasons": value.reasons,
        "rule_id": value.rule_id,
        "observations": [
            {
                "person": o.person,
                "month": o.month,
                "field": o.field,
                "value": _decimal(o.value),
                "lexical_class": o.lexical_class,
                "origin": o.origin,
                "flag": o.flag,
                "quality": o.quality,
                "reasons": o.reasons,
                "raw_fields": [
                    {
                        "name": r.name,
                        "present": r.present,
                        "value": r.value,
                        "input_type": r.input_type,
                    }
                    for r in o.raw_fields
                ],
            }
            for o in value.observations
        ],
    }


def _measurement_digest(households, bank) -> str:
    return _key(
        "sipp-december-measurements/1",
        {
            "households": [
                {
                    "key": h.key,
                    "members": h.members,
                    "weight": _measurement(h.weight),
                    "person_weights": [_measurement(w) for w in h.person_weights],
                    "measurements": [_measurement(v) for v in h.measurements],
                }
                for h in households
            ],
            "bank": [_measurement(v) for v in bank],
        },
    )


def _counts(households) -> dict:
    per_ssuid = Counter(h.key[0] for h in households)
    return {
        "december_people": sum(len(h.members) for h in households),
        "december_residences": len(households),
        "december_ssuids": len(per_ssuid),
        "multi_residence_ssuids": sum(n > 1 for n in per_ssuid.values()),
        "zero_weight_december_people": sum(
            w.value == 0 for h in households for w in h.person_weights
        ),
    }


def _membership(households) -> str:
    return _key("sipp-december-membership/1", [(h.key, h.members) for h in households])


def _profile(measurements) -> dict:
    values = tuple(measurements)
    return {
        "count": len(values),
        "known": sum(v.value is not None for v in values),
        "unknown": sum(v.value is None for v in values),
        "reasons_nonexclusive": dict(
            sorted(Counter(r for v in values for r in set(v.reasons)).items())
        ),
        "lexical_classes_observations": dict(
            sorted(
                Counter(o.lexical_class for v in values for o in v.observations).items()
            )
        ),
        "flags_observations": dict(
            sorted(
                Counter(
                    "missing_or_invalid" if o.flag is None else str(o.flag)
                    for v in values
                    for o in v.observations
                ).items()
            )
        ),
        "quality_observations": dict(
            sorted(Counter(o.quality for v in values for o in v.observations).items())
        ),
        "copy_conflicts": sum("household_copy_conflict" in v.reasons for v in values),
    }


def _weights(measurements) -> dict:
    values = tuple(measurements)
    known = tuple(v.value for v in values if v.value is not None)
    with localcontext(m._ARITHMETIC_CONTEXT):
        total = sum(known, Decimal(0))
    return {
        "count": len(values),
        "known_denominator": len(known),
        "unknown_count": len(values) - len(known),
        "zero_count": sum(v == 0 for v in known),
        "known_weight_sum": _decimal(total),
    }


def _reconstruct(
    snapshot: Path,
    selection: Selection,
    pins: _Pins,
    binding: str,
    contract: dict,
    chunksize: int,
):
    rows, parser = _parse(snapshot, chunksize)
    source = m.SippSource(pins.sha256, pins.metadata_sha256, 2023, 2022)
    full = m.december_households(rows, source=source, complete_membership=True)
    full_counts = _counts(full)
    if (
        pins.expected_counts is not None
        and tuple(full_counts.values()) != pins.expected_counts
    ):
        raise SourceError("FULL_ROSTER_RECONCILIATION")
    full_membership = _membership(full)
    chosen = _selected_ssuids(tuple(h.key for h in full), selection)
    chosen_set = frozenset(chosen)
    del full
    prepared = m._prepare(rows, FIELDS)
    selected = tuple(raw for key, month, raw in prepared if key[0] in chosen_set)
    del rows, prepared
    projection = canonical_json(
        {
            "format": "sipp-december-lexical/1",
            "fields": FIELDS,
            "rows": [tuple(row.values[f] for f in FIELDS) for row in selected],
        }
    )
    if len(projection) > _PAYLOAD_MAX:
        raise SourceError("PROJECTION_TOO_LARGE")
    households = m.december_households(
        selected, source=source, complete_membership=True, copied_fields=_COPIED
    )
    bank = m.december_bank_assets(selected, source=source)
    projection_key = _key(
        "sipp-december-projection/1",
        (binding, selection, _sha(projection), len(projection)),
    )
    receipt = {
        "format": "sipp_december_source_v1",
        "producer_binding": binding,
        "contract": contract,
        "selection": selection,
        "selected_ssuid_digest": _key("sipp-december-ssuid-selection/1", chosen),
        "full_counts": full_counts,
        "selected_counts": _counts(households),
        "parser": parser,
        "full_membership_digest": full_membership,
        "selected_membership_digest": _membership(households),
        "projection": {
            "key": projection_key,
            "sha256": _sha(projection),
            "bytes": len(projection),
        },
        "measurement_digest": _measurement_digest(households, bank),
        "aggregates": {
            "bank": _profile(bank),
            "vehicles": {
                f: _profile(h.measurements[i] for h in households)
                for i, f in enumerate(_COPIED)
            },
            "person_weights": _weights(w for h in households for w in h.person_weights),
            "household_weights": _weights(h.weight for h in households),
        },
    }
    receipt_bytes = canonical_json(receipt)
    if len(receipt_bytes) > _PAYLOAD_MAX:
        raise SourceError("RECEIPT_TOO_LARGE")
    receipt_key = _key(
        "sipp-december-receipt/1", (binding, _sha(receipt_bytes), len(receipt_bytes))
    )
    return households, bank, projection, receipt_bytes, projection_key, receipt_key


def _bounded_object(
    store: ContentStore, key: str, namespace: Path, *, kind: str, binding: str
) -> bytes:
    """Bound two files before letting the generic store inspect its own format."""
    if not _KEY.fullmatch(key):
        raise SourceError("INVALID_ARTIFACT_KEY")
    obj = store.object_path(key)
    for path in (store.root, store.root / "objects", obj.parent, obj):
        _directory(path)
    payload_name = "payload.bin" if kind == "bytes" else "payload.json"
    before = _identity(obj.lstat())

    def names():
        # Two entries are expected; a third suffices to refuse any larger tree.
        with os.scandir(obj) as entries:
            return {entry.name for entry in islice(entries, 3)}

    if names() != {"meta.json", payload_name}:
        raise SourceError("STORE_OBJECT_LAYOUT")
    with tempfile.TemporaryDirectory(prefix="object-", dir=namespace) as directory:
        owned = ContentStore(directory)
        dest = owned.object_path(key)
        dest.mkdir(parents=True)
        _copy_regular(obj / "meta.json", dest / "meta.json", _META_MAX)
        _copy_regular(obj / payload_name, dest / payload_name, _PAYLOAD_MAX)
        if before != _identity(obj.lstat()) or names() != {
            "meta.json",
            payload_name,
        }:
            raise SourceError("STORE_OBJECT_CHANGED")
        metadata = owned.metadata(key, kind=kind)
        if metadata.get("node_key") != binding or set(metadata["payloads"]) != {
            payload_name
        }:
            raise SourceError("STORE_BINDING")
        if kind == "bytes":
            return owned.load_bytes(key)
        owned.load_json(key, kind=kind)
        return (dest / payload_name).read_bytes()


def _publish(store, namespace, key, payload, *, kind, binding):
    if len(payload) > _PAYLOAD_MAX:
        raise SourceError("PAYLOAD_TOO_LARGE")
    _directory(store.root / "objects")
    if os.path.lexists(store.object_path(key).parent):
        _directory(store.object_path(key).parent)
    if os.path.lexists(store.object_path(key)):
        if (
            _bounded_object(store, key, namespace, kind=kind, binding=binding)
            != payload
        ):
            raise SourceError("STORE_CONTENT_COLLISION")
    # Every cooperating writer holds the namespace lock, including this check/write.
    if kind == "bytes":
        store.put_bytes(key, payload, node_key=binding, verify_existing=True)
    else:
        store.put_json(
            key, json.loads(payload), kind=kind, node_key=binding, verify_existing=True
        )
    if _bounded_object(store, key, namespace, kind=kind, binding=binding) != payload:
        raise SourceError("STORE_CONTENT_COLLISION")


def _run(
    parent_path: Path,
    store: ContentStore,
    selection: Selection,
    supplied_receipt: str | None,
) -> SippDecemberSource:
    if type(selection) is not str or selection not in _MODES:
        raise SourceError("UNSUPPORTED_SELECTION")
    if not isinstance(store, ContentStore):
        raise SourceError("INVALID_CONTENT_STORE")
    if supplied_receipt is not None and (
        type(supplied_receipt) is not str or not _KEY.fullmatch(supplied_receipt)
    ):
        raise SourceError("INVALID_ARTIFACT_KEY")
    pins = _PINS
    started = time.monotonic()
    snapshot_dir = None
    try:
        with _locked(store) as namespace:
            resources = _resources(namespace, pins)
            binding, contract = _contract(pins)
            phases = {}
            snapshot_dir = Path(tempfile.mkdtemp(prefix="snapshot-", dir=namespace))
            snapshot = snapshot_dir / "parent.bin"
            phase = time.monotonic()
            digest = _copy_regular(
                Path(parent_path), snapshot, pins.size, exact_size=pins.size
            )
            phases["parent_copy_sha_seconds"] = time.monotonic() - phase
            if digest != pins.sha256:
                raise SourceError("PARENT_SHA256")
            phase = time.monotonic()
            if (
                source_content_key(_SOURCE_NAME, snapshot)
                != contract["parent"]["content_key"]
            ):
                raise SourceError("SNAPSHOT_CONTENT_KEY")
            phases["persisted_snapshot_identity_seconds"] = time.monotonic() - phase
            phase = time.monotonic()
            first = _reconstruct(snapshot, selection, pins, binding, contract, 8192)
            phases["first_parse_measure_seconds"] = time.monotonic() - phase
            if supplied_receipt is None:
                expected = first[2:]
                del first
                phase = time.monotonic()
                reconstructed = _reconstruct(
                    snapshot, selection, pins, binding, contract, 4096
                )
                phases["second_parse_measure_seconds"] = time.monotonic() - phase
                if expected != reconstructed[2:]:
                    raise SourceError("SECOND_PASS_CHANGED")
                del expected
            else:
                reconstructed = first
            households, bank, projection, receipt, projection_key, receipt_key = (
                reconstructed
            )
            if _contract(pins)[0] != binding or _PINS != pins:
                raise SourceError("IMPLEMENTATION_CHANGED")
            phase = time.monotonic()
            if supplied_receipt is None:
                _publish(
                    store,
                    namespace,
                    projection_key,
                    projection,
                    kind="bytes",
                    binding=binding,
                )
                _publish(
                    store,
                    namespace,
                    receipt_key,
                    receipt,
                    kind="sipp_december_source_v1",
                    binding=binding,
                )
            else:
                if supplied_receipt != receipt_key:
                    raise SourceError("RECEIPT_IDENTITY")
                if (
                    _bounded_object(
                        store,
                        supplied_receipt,
                        namespace,
                        kind="sipp_december_source_v1",
                        binding=binding,
                    )
                    != receipt
                ):
                    raise SourceError("RECEIPT_CHANGED")
                if (
                    _bounded_object(
                        store, projection_key, namespace, kind="bytes", binding=binding
                    )
                    != projection
                ):
                    raise SourceError("PROJECTION_CHANGED")
            phases["store_seconds"] = time.monotonic() - phase
            info = canonical_json(
                {
                    "kind": "sipp_december_run_info_v1",
                    "receipt_key": receipt_key,
                    "mode": "produce" if supplied_receipt is None else "read",
                    "wall_time_ns": time.time_ns(),
                    "elapsed_seconds": time.monotonic() - started,
                    "phase_seconds": phases,
                    "resource_preflight": resources,
                    "loaded_microcosm_modules": {
                        name: {
                            "path": module.__file__,
                            "sha256": _sha(Path(module.__file__).read_bytes()),
                        }
                        for name, module in tuple(sorted(sys.modules.items()))
                        if name.startswith("microcosm.")
                        and getattr(module, "__file__", None)
                    },
                    "process_peak_rss_raw": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "process_peak_rss_raw_unit": "bytes"
                    if sys.platform == "darwin"
                    else "KiB",
                    "free_disk_after_verification_before_snapshot_cleanup_bytes": shutil.disk_usage(
                        namespace
                    ).free,
                }
            )
            info_key = _key("sipp-december-run-info/1", (_sha(info), len(info)))
            _publish(
                store,
                namespace,
                info_key,
                info,
                kind="sipp_december_run_info_v1",
                binding=binding,
            )
            shutil.rmtree(snapshot_dir)
            snapshot_dir = None
            return SippDecemberSource(
                households,
                bank,
                receipt,
                receipt_key,
                projection_key,
                binding,
                info_key,
            )
    except (
        OSError,
        UnicodeError,
        csv.Error,
        StoreError,
        m.MeasurementError,
        ValueError,
        TypeError,
        RecursionError,
        ImportError,
        subprocess.SubprocessError,
    ) as error:
        code = (
            str(error)
            if isinstance(error, SourceError)
            else "SOURCE_VERIFICATION_FAILED"
        )
        if snapshot_dir is not None:
            failure = {
                "status": "failed",
                "reason": code,
                "snapshot": snapshot_dir.name,
                "parent_bytes_retained": (snapshot_dir / "parent.bin").stat().st_size
                if (snapshot_dir / "parent.bin").exists()
                else 0,
            }
            try:
                (snapshot_dir / "failure.json").write_bytes(canonical_json(failure))
            except OSError:
                pass
        raise SourceError(code) from None


def produce_sipp_december_source(
    parent_path: str | Path, store: ContentStore, *, selection: Selection
) -> SippDecemberSource:
    """Authenticate the closed parent twice through one owned snapshot and publish.

    Both parses traverse all physical rows; selection retains whole December SSUIDs.
    The result is source evidence only. Genuine execution requires external approval.
    """
    return _run(Path(parent_path), store, selection, None)


def load_sipp_december_source(
    parent_path: str | Path,
    store: ContentStore,
    receipt_key: str,
    *,
    expected_selection: Selection,
) -> SippDecemberSource:
    """Reconstruct from a fresh authenticated parent; never trust a caller receipt."""
    return _run(Path(parent_path), store, expected_selection, receipt_key)
