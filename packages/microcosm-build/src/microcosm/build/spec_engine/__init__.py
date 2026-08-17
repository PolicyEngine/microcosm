"""The Microcosm country specification compiler front end.

F0 stops at resolved configuration and generated compatibility payloads.  It
does not execute stages or construct bundle-mode authority.
"""

from .canonical import (
    CANONICALIZER_ID,
    CANONICALIZER_VERSION,
    canonical_json_bytes,
)
from .errors import (
    SpecEngineError,
    SpecParseError,
    SpecSchemaError,
    SpecValidationError,
)
from .loader import (
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
    SpecBinding,
    Surface,
    SurfaceObjects,
)
from .resolver import (
    LEGACY_V1_PROTOCOL,
    KernelRegistry,
    SeedProtocol,
    SpecResolutionError,
)
from .schemas import SchemaRegistry, load_schema_registry
from .yaml12 import load_yaml12, load_yaml12_file

__all__ = [
    "ArtifactSpec",
    "CANONICALIZER_ID",
    "CANONICALIZER_VERSION",
    "ColumnSpec",
    "EntitySpec",
    "GrammarReceipt",
    "KernelRegistry",
    "LEGACY_V1_PROTOCOL",
    "ResolvedSpec",
    "ResourceDescriptor",
    "ResourceKind",
    "SchemaRegistry",
    "ScopeSpec",
    "SeedProtocol",
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
    "canonical_json_bytes",
    "emit_bundle_lock",
    "load_bundle",
    "load_schema_registry",
    "load_yaml12",
    "load_yaml12_file",
]
