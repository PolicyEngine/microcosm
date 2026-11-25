"""Generated PolicyEngine ABI lock derivation and fail-closed verification.

The bundle owns take-up treatments; the installed rules engine owns the ABI
facts those treatments consume.  This module joins those authorities only to
emit and verify ``engine_abi.lock.json``.  The generated lock is the one exact
runtime/source pin: vintage rows index its version rather than copying it.

Deriving the lock imports policyengine-us, builds two tax-benefit systems and
indexes policyengine-us's source code. A process that passed the full check can
hand the build processes it starts a record of what it compared
(:func:`engine_abi_lock_check_environment`); a child with its parent's record
compares those values instead of deriving the lock again.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .canonical import canonical_json_bytes
from .errors import SpecValidationError
from .schemas import SchemaRegistry, load_schema_registry

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from .model import ResolvedSpec

ENGINE_ABI_LOCK_FILENAME = "engine_abi.lock.json"
ENGINE_ABI_LOCK_SCHEMA_ID = "locks.schema.json#/$defs/engine_abi_lock"
POLICYENGINE_US_PACKAGE = "policyengine-us"
US_TAKE_UP_PROGRAM_COUNT = 17
_ENGINE_VERSION_REF: Mapping[str, str] = {
    "kind": "engine_abi_lock",
    "pointer": "/engine/version",
}
_REMAINING_STAGE_VERSIONED_CONTRACTS = (
    "ssi_dependency_contract",
    "engine_input_projection_contract",
)
_ACTIVE_TAKE_UP_MANIFEST_PROGRAM_BINDINGS: ContextVar[
    tuple[tuple[str, str, str], ...] | None
] = ContextVar("active_take_up_manifest_program_bindings", default=None)
#: The environment variable through which a process that passed the lock check
#: hands a record of that check to the build processes it starts.
ENGINE_ABI_LOCK_CHECK_RECORD_VARIABLE = "MICROCOSM_ENGINE_ABI_LOCK_CHECK_RECORD"
_LOCK_CHECK_RECORD_SCHEMA_VERSION = 1
_LOCK_CHECK_RECORD_KEYS = frozenset(
    {"schema_version", "issuer_pid", "lock_sha256", "engine", "domains_sha256"}
)
_SHA256_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _LockCheckInputs:
    """What one passed lock check compared, which a child process must match.

    Besides microcosm's own code, the fresh derivation reads only the installed
    policyengine-us and the bundle's typed take-up and sources domains.
    Microcosm's code is not recorded: only the issuing process's direct
    children act on a record (others ignore it), and they run the same
    checkout within the same run.
    A source edit during a base run is the base stage's run-config check to
    catch (its builder code digest), not this record's.

    The engine is matched by its installed distribution version only, as the
    base stage's run config records it. A policyengine-us reinstalled with
    different files under the same version during a run, or a changed
    policyengine-core, is not noticed by a child holding a record.
    """

    lock_sha256: str
    engine_version: str
    domains_sha256: str

    def record(self, *, issuer_pid: int) -> dict[str, object]:
        return {
            "schema_version": _LOCK_CHECK_RECORD_SCHEMA_VERSION,
            "issuer_pid": issuer_pid,
            "lock_sha256": self.lock_sha256,
            "engine": {
                "package": POLICYENGINE_US_PACKAGE,
                "version": self.engine_version,
            },
            "domains_sha256": self.domains_sha256,
        }


#: Lock files this process has checked, by path, with what each check compared.
_PASSED_LOCK_CHECKS: dict[str, _LockCheckInputs] = {}


def active_take_up_manifest_program_bindings() -> (
    tuple[tuple[str, str, str], ...] | None
):
    """Return fresh typed bindings during recursive-free lock derivation only."""

    return _ACTIVE_TAKE_UP_MANIFEST_PROGRAM_BINDINGS.get()


@contextmanager
def scoped_take_up_manifest_program_bindings(
    bindings: tuple[tuple[str, str, str], ...],
):
    """Temporarily supply reviewed take-up bindings during bootstrap imports."""

    token = _ACTIVE_TAKE_UP_MANIFEST_PROGRAM_BINDINGS.set(bindings)
    try:
        yield
    finally:
        _ACTIVE_TAKE_UP_MANIFEST_PROGRAM_BINDINGS.reset(token)


def _installed_engine_version(package: str) -> str:
    """Return the installed distribution version (a test seam by design)."""

    return importlib.metadata.version(package)


def _fresh_policyengine_us_contract() -> Mapping[str, Mapping[str, object]]:
    """Derive take-up ABI facts from a fresh installed engine adapter."""

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    return PolicyEngineUSEngine().take_up_contract()


def _fresh_remaining_stage_input_manifest(
    take_up: Mapping[str, object],
    *,
    engine_abi_programs: Mapping[str, object],
    sources_document: Mapping[str, object],
) -> dict[str, object]:
    """Derive the complete post-transfer read manifest from installed code.

    This is deliberately colocated with the engine-lock builder.  The lock is
    generated evidence, so production loading verifies this fresh derivation
    before compilation; authored country YAML never imports or copies the
    runtime registry or either of its digests.
    """

    from microcosm.build.spec_engine.take_up_semantics import (
        project_legacy_take_up_contract,
    )

    projection = project_legacy_take_up_contract(
        take_up,
        engine_abi_lock={"programs": engine_abi_programs},
        sources_document=sources_document,
    )
    raw_programs = projection.get("programs")
    if not isinstance(raw_programs, list):  # pragma: no cover - projector invariant
        raise SpecValidationError("projected take-up programs must be an array")
    take_up_program_bindings: list[tuple[str, str, str]] = []
    for index, value in enumerate(raw_programs):
        row = _require_mapping(
            value,
            location=f"projected take-up programs/{index}",
        )
        variable = row.get("variable")
        entity = row.get("entity")
        treatment = row.get("populace_treatment")
        if not all(
            isinstance(item, str) and item for item in (variable, entity, treatment)
        ):
            raise SpecValidationError(
                "projected take-up manifest binding requires non-empty variable, "
                f"entity, and populace_treatment at index {index}"
            )
        take_up_program_bindings.append((variable, entity, treatment))

    bindings = tuple(take_up_program_bindings)

    with scoped_take_up_manifest_program_bindings(bindings):
        # Importing the legacy runtime eagerly derives static agreement
        # surfaces.  The context binding lets that import consume this same
        # typed/fresh projection instead of re-entering CountrySpec and the
        # lock verifier currently executing.
        from microcosm.build.us_runtime.multispine_pool import (
            pool_remaining_stage_input_manifest,
            pool_remaining_stage_input_manifest_receipt,
        )

        entries = pool_remaining_stage_input_manifest(
            take_up_program_bindings=bindings,
        )
        rows = [
            {
                "stage": entry.stage,
                "consumer": entry.consumer,
                "entity": entry.entity,
                "variable": entry.variable,
                "execution_scope": entry.execution_scope,
                "provision": entry.provision,
                "available_by": entry.available_by,
                "fallback": entry.fallback,
            }
            for entry in entries
        ]
        receipt = dict(
            pool_remaining_stage_input_manifest_receipt(
                take_up_program_bindings=bindings,
            )
        )
    observed_manifest_sha256 = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    if receipt.get("manifest_sha256") != observed_manifest_sha256:
        raise SpecValidationError(
            "fresh remaining-stage input receipt does not bind its normalized rows"
        )
    if receipt.get("entry_count") != len(rows):
        raise SpecValidationError(
            "fresh remaining-stage input receipt has the wrong entry count"
        )
    stage_counts = {
        stage: sum(row["stage"] == stage for row in rows)
        for stage in ("derive", "seed", "simulate")
    }
    consumer_names = sorted({str(row["consumer"]) for row in rows})
    consumer_counts = {
        consumer: sum(row["consumer"] == consumer for row in rows)
        for consumer in consumer_names
    }
    if receipt.get("stage_counts") != stage_counts:
        raise SpecValidationError(
            "fresh remaining-stage input receipt has stale stage counts"
        )
    if receipt.get("consumer_counts") != consumer_counts:
        raise SpecValidationError(
            "fresh remaining-stage input receipt has stale consumer counts"
        )
    receipt_body = {key: value for key, value in receipt.items() if key != "sha256"}
    observed_receipt_sha256 = hashlib.sha256(
        canonical_json_bytes(receipt_body)
    ).hexdigest()
    if receipt.get("sha256") != observed_receipt_sha256:
        raise SpecValidationError(
            "fresh remaining-stage input receipt does not bind its body"
        )
    return {"rows": rows, "receipt": receipt}


def _require_mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _normalize_remaining_stage_engine_version_refs(
    manifest: Mapping[str, object],
    *,
    engine_version: str,
) -> dict[str, object]:
    """Replace verified legacy version copies with the lock's sole typed ref.

    The runtime receipt is intentionally validated in its generation-0 shape
    before this function is called.  Its SHA fields therefore remain exact
    compatibility identities for that legacy shape, while the generated lock
    stores the engine version itself only once at ``/engine/version``.
    """

    normalized = dict(manifest)
    receipt = dict(
        _require_mapping(
            manifest.get("receipt"),
            location="fresh remaining-stage input manifest/receipt",
        )
    )
    for contract_name in _REMAINING_STAGE_VERSIONED_CONTRACTS:
        contract = dict(
            _require_mapping(
                receipt.get(contract_name),
                location=(
                    f"fresh remaining-stage input manifest/receipt/{contract_name}"
                ),
            )
        )
        observed_version = contract.pop("engine_version", None)
        if observed_version != engine_version:
            raise SpecValidationError(
                "fresh remaining-stage input receipt engine version differs "
                "from the exact generated engine pin: "
                f"{contract_name}={observed_version!r}, "
                f"engine/version={engine_version!r}"
            )
        if "engine_version_ref" in contract:
            raise SpecValidationError(
                "fresh remaining-stage input receipt must be validated in its "
                f"legacy shape before normalization: {contract_name} already "
                "contains engine_version_ref"
            )
        contract["engine_version_ref"] = dict(_ENGINE_VERSION_REF)
        receipt[contract_name] = contract
    normalized["receipt"] = receipt
    return normalized


def _take_up_program_bindings(take_up: Mapping[str, object]) -> dict[str, str]:
    programs = take_up.get("programs")
    if not isinstance(programs, list):
        raise SpecValidationError("take_up/programs: array required")
    if len(programs) != US_TAKE_UP_PROGRAM_COUNT:
        raise SpecValidationError(
            "take_up/programs: engine ABI mapping must cover exactly "
            f"{US_TAKE_UP_PROGRAM_COUNT} programs; found {len(programs)}"
        )

    bindings: dict[str, str] = {}
    variables: list[str] = []
    for index, raw in enumerate(programs):
        row = _require_mapping(raw, location=f"take_up/programs/{index}")
        program_id = row.get("id")
        variable = row.get("variable")
        if not isinstance(program_id, str) or not program_id:
            raise SpecValidationError(
                f"take_up/programs/{index}/id: non-empty string required"
            )
        if not isinstance(variable, str) or not variable:
            raise SpecValidationError(
                f"take_up/programs/{index}/variable: non-empty string required"
            )
        if program_id in bindings:
            raise SpecValidationError(
                f"take_up/programs: duplicate program id {program_id!r}"
            )
        bindings[program_id] = variable
        variables.append(variable)

    duplicate_variables = sorted(
        {variable for variable in variables if variables.count(variable) > 1}
    )
    if duplicate_variables:
        raise SpecValidationError(
            "take_up/programs: program-to-variable mapping must be injective; "
            f"duplicates={duplicate_variables!r}"
        )
    return bindings


def engine_abi_lock_payload_from_domains(
    domains: Mapping[str, object],
) -> dict[str, object]:
    """Derive the lock from normalized take-up plus the fresh installed engine.

    Program ids and treatments come from the bundle; the exact distribution
    version, variable metadata, and consumers come only from the installed
    engine.  No authored YAML literal participates in this derivation.
    """

    take_up = _require_mapping(domains.get("take_up"), location="take_up")
    sources = _require_mapping(domains.get("sources"), location="sources")
    package = POLICYENGINE_US_PACKAGE
    installed_version = _installed_engine_version(package)
    if not installed_version:
        raise SpecValidationError("installed engine version must be non-empty")

    bindings = _take_up_program_bindings(take_up)
    engine_contract = _fresh_policyengine_us_contract()
    bundle_variables = set(bindings.values())
    engine_variables = set(engine_contract)
    if bundle_variables != engine_variables:
        raise SpecValidationError(
            "take_up/programs: mapping is not total over the fresh engine ABI; "
            f"missing={sorted(engine_variables - bundle_variables)!r}, "
            f"extra={sorted(bundle_variables - engine_variables)!r}"
        )

    programs: dict[str, object] = {}
    for program_id, variable in sorted(bindings.items()):
        facts = _require_mapping(
            engine_contract[variable],
            location=f"fresh engine ABI/{variable}",
        )
        value_type = facts.get("value_type")
        if value_type not in {"bool", "float", "int"}:
            raise SpecValidationError(
                f"fresh engine ABI/{variable}/value_type: unsupported {value_type!r}"
            )
        consumers = facts.get("consumers")
        if not isinstance(consumers, list) or not all(
            isinstance(consumer, str) and consumer for consumer in consumers
        ):
            raise SpecValidationError(
                f"fresh engine ABI/{variable}/consumers: string array required"
            )
        entity = facts.get("entity")
        engine_class = facts.get("engine_class")
        if not isinstance(entity, str) or not entity:
            raise SpecValidationError(
                f"fresh engine ABI/{variable}/entity: non-empty string required"
            )
        if not isinstance(engine_class, str) or not engine_class:
            raise SpecValidationError(
                f"fresh engine ABI/{variable}/engine_class: non-empty string required"
            )
        programs[program_id] = {
            "variable": variable,
            "entity": entity,
            "value_type": value_type,
            "default": facts.get("default"),
            "engine_class": engine_class,
            "consumers": sorted(consumers),
        }

    remaining_stage_input_manifest = _fresh_remaining_stage_input_manifest(
        take_up,
        engine_abi_programs=programs,
        sources_document=sources,
    )
    normalized_remaining_stage_input_manifest = (
        _normalize_remaining_stage_engine_version_refs(
            remaining_stage_input_manifest,
            engine_version=installed_version,
        )
    )
    return {
        "engine": {"package": package, "version": installed_version},
        "programs": programs,
        "remaining_stage_input_manifest": normalized_remaining_stage_input_manifest,
    }


def engine_abi_lock_bytes_from_domains(domains: Mapping[str, object]) -> bytes:
    """Return canonical JSON plus the repository's deterministic final LF."""

    payload = engine_abi_lock_payload_from_domains(domains)
    load_schema_registry().validate(payload, ENGINE_ABI_LOCK_SCHEMA_ID)
    return canonical_json_bytes(payload) + b"\n"


def engine_abi_lock_payload(spec: ResolvedSpec) -> dict[str, object]:
    """Derive a fresh lock from typed take-up plus the installed engine."""

    return engine_abi_lock_payload_from_domains(
        {
            "take_up": spec.domain("take_up").to_wire(),
            "sources": spec.domain("sources").to_wire(),
        }
    )


def engine_abi_lock_bytes(spec: ResolvedSpec) -> bytes:
    """Return canonical lock JSON plus the repository's deterministic LF."""

    payload = engine_abi_lock_payload(spec)
    load_schema_registry().validate(payload, ENGINE_ABI_LOCK_SCHEMA_ID)
    return canonical_json_bytes(payload) + b"\n"


def emit_engine_abi_lock(spec: ResolvedSpec, path: str | Path) -> Path:
    """Emit the generated lock; callers must not add it to the manifest."""

    destination = Path(path)
    destination.write_bytes(engine_abi_lock_bytes(spec))
    return destination


def assert_engine_abi_lock_current(
    root: Traversable | Path,
    domains: Mapping[str, object],
    *,
    schema_registry: SchemaRegistry,
) -> Mapping[str, object] | None:
    """Refuse a missing, malformed, hand-edited, or stale generated lock.

    Bundles without typed take-up do not need an ABI lock.  A lock with no
    take-up surface is refused because it has no bundle mapping to attest.

    A process whose parent passed this check for the same lock and handed it a
    record (:func:`engine_abi_lock_check_environment`) compares the lock file's
    SHA-256, the installed policyengine-us version and the take-up and sources
    domains with that record instead of deriving the lock again, and refuses
    to run when any of them differs.
    """

    take_up = domains.get("take_up")
    lock_resource = root.joinpath(*PurePosixPath(ENGINE_ABI_LOCK_FILENAME).parts)
    if take_up is None:
        if lock_resource.is_file():
            raise SpecValidationError(
                "engine_abi.lock.json is present without a typed take-up surface"
            )
        return None

    parent_check = _parent_lock_check()
    if parent_check is not None:
        parsed = _lock_matching_parent_check(
            lock_resource,
            domains,
            parent_check,
            schema_registry=schema_registry,
        )
        _PASSED_LOCK_CHECKS[str(lock_resource)] = parent_check
        return parsed

    try:
        expected_payload = engine_abi_lock_payload_from_domains(domains)
    except importlib.metadata.PackageNotFoundError:
        expected_payload = None
    if expected_payload is None:
        # The engine distribution is absent, so currency is unattestable in
        # this environment (e.g. the wheels gate's engine-less venv). The
        # committed lock is still validated structurally — present, schema-
        # valid, canonical UTF-8 JSON in canonical byte form — and returned
        # unattested. Every environment that regenerates the lock or builds
        # against engine facts installs the engine and takes the attested
        # path below, which stays fail-closed on any drift.
        raw, parsed = _read_schema_valid_lock(lock_resource, schema_registry)
        if raw != canonical_json_bytes(parsed) + b"\n":
            raise SpecValidationError(
                "generated lock is not in canonical byte form",
                source=ENGINE_ABI_LOCK_FILENAME,
            )
        return parsed
    schema_registry.validate(expected_payload, ENGINE_ABI_LOCK_SCHEMA_ID)
    expected_bytes = canonical_json_bytes(expected_payload) + b"\n"
    raw, parsed = _read_schema_valid_lock(lock_resource, schema_registry)
    if raw != expected_bytes:
        raise SpecValidationError(
            "generated lock is stale or non-canonical; regenerate it from the "
            "fresh installed engine ABI",
            source=ENGINE_ABI_LOCK_FILENAME,
        )
    engine = _require_mapping(expected_payload["engine"], location="engine")
    _PASSED_LOCK_CHECKS[str(lock_resource)] = _LockCheckInputs(
        lock_sha256=hashlib.sha256(raw).hexdigest(),
        engine_version=str(engine["version"]),
        domains_sha256=_lock_check_domains_sha256(domains),
    )
    return parsed


def engine_abi_lock_check_environment(root: Traversable | Path) -> dict[str, str]:
    """Environment entries that hand this process's lock check to a child.

    Returns the record of the check this process passed for ``root``'s lock
    file (fully, or by matching its own parent's record), bound to this
    process's ID, or nothing when no such check passed here.  A child honours
    the record only when its parent process issued it, so a record inherited
    from further up the process tree is ignored and that child runs the full
    check.
    """

    lock_resource = root.joinpath(*PurePosixPath(ENGINE_ABI_LOCK_FILENAME).parts)
    passed = _PASSED_LOCK_CHECKS.get(str(lock_resource))
    if passed is None:
        return {}
    return {
        ENGINE_ABI_LOCK_CHECK_RECORD_VARIABLE: json.dumps(
            passed.record(issuer_pid=os.getpid()),
            sort_keys=True,
            separators=(",", ":"),
        )
    }


def _read_schema_valid_lock(
    lock_resource: Traversable | Path,
    schema_registry: SchemaRegistry,
) -> tuple[bytes, Mapping[str, object]]:
    if not lock_resource.is_file():
        raise SpecValidationError(
            "engine_abi.lock.json is required for a bundle with a policy-engine ABI"
        )
    try:
        raw = lock_resource.read_bytes()
    except OSError as error:
        raise SpecValidationError(
            f"unable to read generated lock: {error}",
            source=ENGINE_ABI_LOCK_FILENAME,
        ) from error
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SpecValidationError(
            "generated lock is not canonical UTF-8 JSON",
            source=ENGINE_ABI_LOCK_FILENAME,
        ) from error
    schema_registry.validate(parsed, ENGINE_ABI_LOCK_SCHEMA_ID)
    assert isinstance(parsed, Mapping)
    return raw, parsed


def _lock_check_domains_sha256(domains: Mapping[str, object]) -> str:
    """Digest the two bundle domains the fresh lock derivation reads."""

    take_up = _require_mapping(domains.get("take_up"), location="take_up")
    sources = _require_mapping(domains.get("sources"), location="sources")
    return hashlib.sha256(
        canonical_json_bytes({"take_up": take_up, "sources": sources})
    ).hexdigest()


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX_DIGITS
    )


def _parent_lock_check() -> _LockCheckInputs | None:
    """Read the lock check record this process's parent handed it, if any."""

    raw = os.environ.get(ENGINE_ABI_LOCK_CHECK_RECORD_VARIABLE)
    if raw is None:
        return None

    def malformed(reason: str) -> SpecValidationError:
        return SpecValidationError(
            f"{ENGINE_ABI_LOCK_CHECK_RECORD_VARIABLE} holds a malformed lock "
            f"check record: {reason}; refusing to run",
            source=ENGINE_ABI_LOCK_FILENAME,
        )

    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise malformed("it is not JSON") from error
    if not isinstance(record, dict) or set(record) != _LOCK_CHECK_RECORD_KEYS:
        raise malformed(
            f"it must be an object with keys {sorted(_LOCK_CHECK_RECORD_KEYS)}"
        )
    if record["schema_version"] != _LOCK_CHECK_RECORD_SCHEMA_VERSION:
        raise malformed(f"schema_version {record['schema_version']!r} is not 1")
    issuer_pid = record["issuer_pid"]
    if (
        isinstance(issuer_pid, bool)
        or not isinstance(issuer_pid, int)
        or issuer_pid < 1
    ):
        raise malformed("issuer_pid must be a positive process ID")
    for field in ("lock_sha256", "domains_sha256"):
        if not _is_sha256_hex(record[field]):
            raise malformed(f"{field} must be a lowercase SHA-256 hex digest")
    engine = record["engine"]
    if (
        not isinstance(engine, dict)
        or set(engine) != {"package", "version"}
        or engine["package"] != POLICYENGINE_US_PACKAGE
        or not isinstance(engine["version"], str)
        or not engine["version"]
    ):
        raise malformed(
            f"engine must name package {POLICYENGINE_US_PACKAGE!r} and a version"
        )
    if issuer_pid != os.getppid():
        # Issued further up the process tree, not by this process's parent.
        return None
    return _LockCheckInputs(
        lock_sha256=record["lock_sha256"],
        engine_version=engine["version"],
        domains_sha256=record["domains_sha256"],
    )


def _lock_matching_parent_check(
    lock_resource: Traversable | Path,
    domains: Mapping[str, object],
    parent_check: _LockCheckInputs,
    *,
    schema_registry: SchemaRegistry,
) -> Mapping[str, object]:
    """Return the lock if it and its inputs are those the parent checked."""

    raw, parsed = _read_schema_valid_lock(lock_resource, schema_registry)
    observed_lock_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_lock_sha256 != parent_check.lock_sha256:
        raise SpecValidationError(
            "the lock file differs from the one the parent "
            f"process checked (SHA-256 {observed_lock_sha256}, parent checked "
            f"{parent_check.lock_sha256}); refusing to run",
            source=ENGINE_ABI_LOCK_FILENAME,
        )
    try:
        installed_version: str | None = _installed_engine_version(
            POLICYENGINE_US_PACKAGE
        )
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if installed_version != parent_check.engine_version:
        observed = "not installed" if installed_version is None else installed_version
        raise SpecValidationError(
            "the installed policyengine-us differs from "
            "the one the parent process checked the lock file against "
            f"({observed}, parent checked {parent_check.engine_version}); "
            "refusing to run",
            source=ENGINE_ABI_LOCK_FILENAME,
        )
    observed_domains_sha256 = _lock_check_domains_sha256(domains)
    if observed_domains_sha256 != parent_check.domains_sha256:
        raise SpecValidationError(
            "the take-up and sources domains differ from "
            "those the parent process checked the lock file against (SHA-256 "
            f"{observed_domains_sha256}, parent checked "
            f"{parent_check.domains_sha256}); refusing to run",
            source=ENGINE_ABI_LOCK_FILENAME,
        )
    if parsed.get("engine") != {
        "package": POLICYENGINE_US_PACKAGE,
        "version": parent_check.engine_version,
    }:
        raise SpecValidationError(
            "the lock file does not name the policyengine-us "
            "version the parent process checked it against; refusing to run",
            source=ENGINE_ABI_LOCK_FILENAME,
        )
    return parsed


__all__ = [
    "ENGINE_ABI_LOCK_CHECK_RECORD_VARIABLE",
    "ENGINE_ABI_LOCK_FILENAME",
    "ENGINE_ABI_LOCK_SCHEMA_ID",
    "assert_engine_abi_lock_current",
    "emit_engine_abi_lock",
    "engine_abi_lock_check_environment",
    "engine_abi_lock_bytes",
    "engine_abi_lock_bytes_from_domains",
    "engine_abi_lock_payload",
    "engine_abi_lock_payload_from_domains",
    "scoped_take_up_manifest_program_bindings",
]
