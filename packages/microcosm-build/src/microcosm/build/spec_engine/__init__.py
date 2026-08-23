"""The Microcosm country specification compiler front end.

F0 stops at resolved configuration and generated compatibility payloads.  It
does not execute stages or construct bundle-mode authority.
"""

from .calibration_semantics import (
    CALIBRATION_SUMMARY_ALIASES,
    derive_calibration_summary_aliases,
    project_legacy_calibration_contract,
    resolve_calibration_tail_contracts,
)
from .canonical import (
    CANONICALIZER_ID,
    CANONICALIZER_VERSION,
    canonical_json_bytes,
)
from .compiler_ir import (
    CompiledSpecIR,
    CompilerIRError,
    compile_spec,
)
from .engine_abi import (
    ENGINE_ABI_LOCK_FILENAME,
    assert_engine_abi_lock_current,
    emit_engine_abi_lock,
    engine_abi_lock_bytes,
    engine_abi_lock_bytes_from_domains,
    engine_abi_lock_payload,
    engine_abi_lock_payload_from_domains,
    scoped_take_up_manifest_program_bindings,
)
from .errors import (
    SpecEngineError,
    SpecParseError,
    SpecSchemaError,
    SpecValidationError,
)
from .legacy_adapter import (
    LegacyFieldDiff,
    LegacyPayloadMismatchError,
    assert_legacy_payload_equal,
    compile_to_legacy_payload,
    diff_legacy_payloads,
)
from .loader import (
    GENERATED_LOCK_FILENAMES,
    bundle_lock_bytes,
    bundle_lock_payload,
    emit_bundle_lock,
    load_bundle,
)
from .model import (
    ArtifactSpec,
    ColumnSpec,
    EntitySpec,
    GrammarReceipt,
    ResolvedSpec,
    ResourceDescriptor,
    ResourceKind,
    ScopeSpec,
    SeedSiteBinding,
    SeedSiteOwner,
    SeedSiteOwnerKind,
    SpecBinding,
    Surface,
    SurfaceObjects,
)
from .plan_lock import (
    PlanLockError,
    assert_plan_lock_current,
    emit_plan_lock,
    plan_lock_bytes,
    plan_lock_payload,
)
from .reemission import (
    canonical_yaml_bytes,
    emit_resolved_bundle,
    resolved_bundle_bytes,
)
from .resolver import (
    F0_KERNEL_IDS,
    F0_KERNEL_REGISTRY,
    KernelRegistry,
    SpecResolutionError,
)
from .schemas import SchemaRegistry, load_schema_registry
from .seeds import (
    LEGACY_V1_PROTOCOL,
    DrawSiteProtocol,
    KernelAttestation,
    SeedProtocol,
    validate_seed_protocol_wire,
)
from .take_up_semantics import (
    project_legacy_take_up_contract,
    project_legacy_take_up_identity,
    validate_take_up_semantics,
)
from .yaml12 import load_yaml12, load_yaml12_file

__all__ = [
    "ArtifactSpec",
    "CANONICALIZER_ID",
    "CANONICALIZER_VERSION",
    "CALIBRATION_SUMMARY_ALIASES",
    "ColumnSpec",
    "CompiledSpecIR",
    "CompilerIRError",
    "DrawSiteProtocol",
    "EntitySpec",
    "ENGINE_ABI_LOCK_FILENAME",
    "F0_KERNEL_IDS",
    "F0_KERNEL_REGISTRY",
    "GrammarReceipt",
    "GENERATED_LOCK_FILENAMES",
    "KernelRegistry",
    "KernelAttestation",
    "LEGACY_V1_PROTOCOL",
    "LegacyFieldDiff",
    "LegacyPayloadMismatchError",
    "PlanLockError",
    "ResolvedSpec",
    "ResourceDescriptor",
    "ResourceKind",
    "SchemaRegistry",
    "ScopeSpec",
    "SeedProtocol",
    "SeedSiteBinding",
    "SeedSiteOwner",
    "SeedSiteOwnerKind",
    "SpecBinding",
    "SpecEngineError",
    "SpecParseError",
    "SpecResolutionError",
    "SpecSchemaError",
    "SpecValidationError",
    "Surface",
    "SurfaceObjects",
    "bundle_lock_bytes",
    "bundle_lock_payload",
    "canonical_yaml_bytes",
    "canonical_json_bytes",
    "compile_spec",
    "compile_to_legacy_payload",
    "diff_legacy_payloads",
    "derive_calibration_summary_aliases",
    "assert_engine_abi_lock_current",
    "emit_engine_abi_lock",
    "emit_bundle_lock",
    "emit_plan_lock",
    "emit_resolved_bundle",
    "engine_abi_lock_bytes",
    "engine_abi_lock_bytes_from_domains",
    "engine_abi_lock_payload",
    "engine_abi_lock_payload_from_domains",
    "scoped_take_up_manifest_program_bindings",
    "load_bundle",
    "load_schema_registry",
    "load_yaml12",
    "load_yaml12_file",
    "project_legacy_take_up_contract",
    "project_legacy_take_up_identity",
    "project_legacy_calibration_contract",
    "resolve_calibration_tail_contracts",
    "resolved_bundle_bytes",
    "assert_legacy_payload_equal",
    "assert_plan_lock_current",
    "plan_lock_bytes",
    "plan_lock_payload",
    "validate_take_up_semantics",
    "validate_seed_protocol_wire",
]
