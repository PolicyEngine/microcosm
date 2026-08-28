"""Exact normative-field usage ledger for the F0 compiler front end.

The ledger separates four kinds of use from the question of whether a field
changes generation-0 execution.  In particular, hashing or retaining a field
in a compiled surface is evidence only for an explicitly reviewed
``identity_only`` route; it is never accepted as semantic or legacy-adapter
evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from .canonical import sha256_json, spec_envelope
from .compiler_ir import CompiledSpecIR, compile_spec
from .errors import SpecValidationError
from .legacy_adapter import compile_to_legacy_payload, diff_legacy_payloads
from .model import FrozenValue, ResolvedSpec, ResourceKind, thaw_json
from .resolver import (
    _validate_imputation_concept_coverage,
    _validate_imputation_structure,
)
from .schemas import load_schema_registry

EXPECTED_AUTHORED_FIELD_COUNT = 32_528
EXPECTED_RESOLVED_BINDING_FIELD_COUNT = 10_017
EXPECTED_CONFIGURATION_FIELD_COUNT = 42_545


class FieldUsageError(AssertionError):
    """A normative field has no exact, current compiler disposition."""


class UsageMode(StrEnum):
    """The primary front-end operation that consumes one field."""

    LEGACY_BEHAVIOR = "legacy_behavior"
    COMPILER_SEMANTIC = "compiler_semantic"
    FRONT_END_VALIDATION = "front_end_validation"
    IDENTITY_ONLY = "identity_only"


class Generation0Effect(StrEnum):
    """Whether the field reaches the constants-era compatibility payload."""

    LEGACY_BEHAVIOR = "legacy_behavior"
    NO_GENERATION0_EFFECT = "no_generation0_effect"


@dataclass(frozen=True, slots=True)
class UsageClaim:
    """One reviewed, closed set of terminal configuration pointers."""

    id: str
    source_prefix: str
    mode: UsageMode
    generation0_effect: Generation0Effect
    consumer: str
    verifier: str
    expected_pointer_count: int
    expected_pointer_sha256: str
    legacy_sinks: tuple[str, ...] = ()
    relative_sink_prefix: str | None = None
    rationale: str | None = None
    pointer_class: str = "all"


@dataclass(frozen=True, slots=True)
class FieldUse:
    """One exact source pointer and its reviewed primary disposition."""

    pointer: str
    value_sha256: str
    claim_id: str
    mode: UsageMode
    generation0_effect: Generation0Effect
    consumer: str
    sink_pointers: tuple[str, ...]
    proof: str

    def to_wire(self) -> dict[str, object]:
        return {
            "pointer": self.pointer,
            "value_sha256": self.value_sha256,
            "claim_id": self.claim_id,
            "mode": self.mode.value,
            "generation0_effect": self.generation0_effect.value,
            "consumer": self.consumer,
            "sink_pointers": list(self.sink_pointers),
            "proof": self.proof,
        }


@dataclass(frozen=True, slots=True)
class ClaimReceipt:
    """The exact expanded pointer-set attested by one source claim."""

    id: str
    pointer_count: int
    pointer_sha256: str

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "pointer_count": self.pointer_count,
            "pointer_sha256": self.pointer_sha256,
        }


@dataclass(frozen=True, slots=True)
class FieldUsageLedger:
    """Complete and deterministic exact-pointer usage result."""

    fields: tuple[FieldUse, ...]
    claims: tuple[ClaimReceipt, ...]

    @property
    def source_counts(self) -> dict[str, int]:
        return {
            "authored": sum(
                field.pointer.startswith("/authored/") for field in self.fields
            ),
            "resolved_bindings": sum(
                field.pointer.startswith("/resolved/") for field in self.fields
            ),
        }

    @property
    def mode_counts(self) -> dict[str, int]:
        counts = Counter(field.mode.value for field in self.fields)
        return {mode.value: counts[mode.value] for mode in UsageMode}

    @property
    def generation0_effect_counts(self) -> dict[str, int]:
        counts = Counter(field.generation0_effect.value for field in self.fields)
        return {effect.value: counts[effect.value] for effect in Generation0Effect}

    def field(self, pointer: str) -> FieldUse:
        matches = [field for field in self.fields if field.pointer == pointer]
        if len(matches) != 1:
            raise KeyError(pointer)
        return matches[0]

    def to_wire(self) -> dict[str, object]:
        return {
            "field_count": len(self.fields),
            "source_counts": self.source_counts,
            "mode_counts": self.mode_counts,
            "generation0_effect_counts": self.generation0_effect_counts,
            "pointer_inventory_sha256": sha256_json(
                [field.pointer for field in self.fields]
            ),
            "claims": [claim.to_wire() for claim in self.claims],
            "fields": [field.to_wire() for field in self.fields],
        }


def _wire(value: FrozenValue) -> object:
    return thaw_json(value)


def _pointer(parent: str, token: object) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def _terminal_rows(value: object, *, path: str) -> list[tuple[str, object]]:
    if isinstance(value, Mapping):
        if not value:
            return [(path, {})]
        return [
            row
            for key in sorted(value, key=str)
            for row in _terminal_rows(value[key], path=_pointer(path, key))
        ]
    if isinstance(value, (list, tuple)):
        if not value:
            return [(path, [])]
        return [
            row
            for index, item in enumerate(value)
            for row in _terminal_rows(item, path=_pointer(path, index))
        ]
    return [(path, value)]


def _claim_rows(
    claim: UsageClaim,
    subtree: object,
) -> list[tuple[str, object]]:
    rows = _terminal_rows(subtree, path=claim.source_prefix)
    if claim.pointer_class == "all":
        return rows

    def is_concept_validation(pointer: str) -> bool:
        segments = pointer.split("/")
        return "requires_concepts" in segments or "waiver" in segments

    if claim.pointer_class == "family_execution":
        return [row for row in rows if not is_concept_validation(row[0])]
    if claim.pointer_class == "family_concept_validation":
        return [row for row in rows if is_concept_validation(row[0])]

    def is_stacked_geography_source_identity(pointer: str) -> bool:
        if not isinstance(subtree, (list, tuple)):
            return False
        relative = pointer.removeprefix(claim.source_prefix).removeprefix("/")
        tokens = relative.split("/")
        if len(tokens) < 2:
            return False
        try:
            source_index = int(tokens[0])
        except ValueError:
            return False
        if source_index < 0 or source_index >= len(subtree):
            return False
        source = subtree[source_index]
        if not isinstance(source, Mapping):
            return False
        source_id = source.get("id")
        vintage_id = {
            "us_puma_ladder_2020": "cd_119",
            "us_congressional_district_vintage_crosswalk_117_to_119": "cd_117",
        }.get(source_id)
        if vintage_id is None:
            return False
        if len(tokens) == 2:
            return tokens[1] in {"id", "sha256"}
        if len(tokens) != 4 or tokens[1] != "vintage_authorities":
            return False
        try:
            vintage_index = int(tokens[2])
        except ValueError:
            return False
        authorities = source.get("vintage_authorities")
        if not isinstance(authorities, (list, tuple)) or not (
            0 <= vintage_index < len(authorities)
        ):
            return False
        authority = authorities[vintage_index]
        return (
            isinstance(authority, Mapping)
            and authority.get("id") == vintage_id
            and tokens[3] in {"id", "value"}
        )

    if claim.pointer_class == "stacked_geography_source_identity":
        return [row for row in rows if is_stacked_geography_source_identity(row[0])]
    if claim.pointer_class == "source_validation":
        return [row for row in rows if not is_stacked_geography_source_identity(row[0])]
    raise FieldUsageError(f"{claim.id}: unknown pointer class {claim.pointer_class!r}")


def _pointer_value(value: object, pointer: str) -> tuple[bool, object]:
    current = value
    if pointer == "/":
        return True, current
    for raw_token in pointer.removeprefix("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                return False, None
            current = current[token]
        elif isinstance(current, (list, tuple)):
            try:
                index = int(token)
            except ValueError:
                return False, None
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
        else:
            return False, None
    return True, current


def _semantic_subset(source: object, sink: object) -> bool:
    if isinstance(source, Mapping):
        return (
            isinstance(sink, Mapping)
            and set(source) <= set(sink)
            and all(_semantic_subset(source[key], sink[key]) for key in source)
        )
    if isinstance(source, (list, tuple)):
        return (
            isinstance(sink, (list, tuple))
            and len(source) == len(sink)
            and all(
                _semantic_subset(left, right)
                for left, right in zip(source, sink, strict=True)
            )
        )
    if isinstance(source, bool) or isinstance(sink, bool):
        return type(source) is type(sink) and source == sink
    if isinstance(source, (int, float)) and isinstance(sink, (int, float)):
        return float(source) == float(sink)
    return type(source) is type(sink) and source == sink


def configuration_sources(spec: ResolvedSpec) -> dict[str, object]:
    """Return the exact resolved inputs to the normative spec envelope."""

    return {
        "authored": _wire(spec.surfaces.normative),
        "resolved": {
            "generated_authorities": _wire(spec.generated_authorities),
            "seed_protocol": spec.seed_protocol.to_wire(),
            "seed_site_bindings": [
                binding.to_wire() for binding in spec.seed_site_bindings
            ],
            "vintage_authorities": _wire(spec.vintage_authorities),
        },
    }


def _spec_hash(spec: ResolvedSpec, sources: Mapping[str, object]) -> str:
    authored = sources["authored"]
    resolved = sources["resolved"]
    assert isinstance(authored, Mapping)
    assert isinstance(resolved, Mapping)
    return sha256_json(
        spec_envelope(
            country=spec.country,
            schema_version=spec.schema_version,
            normative_files=authored,
            resolved_bindings=resolved,
        )
    )


def _path_inventory(rows: Sequence[tuple[str, object]]) -> tuple[int, str]:
    paths = [path for path, _ in rows]
    return len(paths), sha256_json(paths)


# Filled with reviewed values below.  Keeping them separate makes an accidental
# wildcard broadening a source diff rather than an automatically accepted field.
_PINS: dict[str, tuple[int, str]] = {
    "battery": (
        797,
        "55c2c0cd2d652216ed35f2d2667f52fe00229eb4336507bef9758397a1e24a17",
    ),
    "bundle_country": (
        1,
        "d40841674279a8854f8eadd926c5da31938781c11fd5c7a1c156954abcae8324",
    ),
    "bundle_dataset_run": (
        1,
        "3b26ea9597d0102ed21685f0c9139a266c133dedb3b2fb24ac636cecc8796102",
    ),
    "bundle_identity_generation": (
        1,
        "6563e9af227416e178f16e0dab644391497fe9f0db4496d72898bd2954e3338c",
    ),
    "bundle_seed_protocol": (
        1,
        "36564747627fb0058bb99e8e8c4397e41d8ecc472a20c49d4b6d7df1003a80ec",
    ),
    "calibration": (
        312,
        "9bad55b4945af1c4213c510942e7b9d22d104bf75b0d9804e4a183e1656c8393",
    ),
    "catalog_columns": (
        1_673,
        "a983c0d68e980a31bd1ad41e7ef3c0cb3db9a7ee4d73454ae46f81eb6d1bb427",
    ),
    "catalog_metadata_waivers": (
        5,
        "64182b6be1ea6d95bff345b30a2aa046b6fa7e8ee61a282b225dfa49c28fbfdc",
    ),
    "country_manifest": (
        98,
        "cbbda6d2d245f04325c0b5a7b986cb71d24d6e3c81a7a3af1544de7f75be2a1f",
    ),
    "generated_authorities": (
        8_606,
        "3f20975597d93f7313583a944eeb9d6437651c4ff20e67628bf6bf4c5aa9f004",
    ),
    "geography_assignment": (
        28,
        "1eb3eaedce22de09799d9a23cc9034f0c048d7b5cc59d76b96b83c5bb49e1508",
    ),
    "geography_phase": (
        1,
        "e6f1851db8754554dcbab4b09ca911a63ddba82fdd9743a7af542f216037bf9d",
    ),
    "imputation_chaining": (
        20,
        "e910767eb8a7df05d22ec1a901a00580a3cd5aa26b66bf881b3297e47cca66b8",
    ),
    "imputation_concepts": (
        20,
        "18a2b6f4cd5c6296092e606243d7b7fb2f00fa513dfd7936aafdbb202407b43d",
    ),
    "imputation_family_concept_validation": (
        52,
        "bdf40812604e7cd35d68093fe14b7cd1371cb8ab8658fd6002b254610b062781",
    ),
    "imputation_family_execution": (
        1_769,
        "cd395f41c1cc425c734dc344607de043f3506b4ec2e727a29130fc4304f351e3",
    ),
    "imputation_gap_fill_schedule": (
        5,
        "deb4acc6f962df9eb458e4206e1a75a3fe596a9a9cdb9c9163e75fd8d4f7b77a",
    ),
    "imputation_models": (
        96,
        "e4d6b6b747fcec1c027e0f1c2d1905274c0426217a61383b02e75baadb93db4d",
    ),
    "imputation_predictor_blocks": (
        62,
        "6cabbe863daed029f8695abe558a23f7dcc465bcb65c88a41cd131725c952bdd",
    ),
    "imputation_primary_checkpoint": (
        2,
        "e1dd7dc5123ab0f39d08ea4939d98dd09a6fdb8e7449a7ca3125fb1ddbd5b4e9",
    ),
    "imputation_producer_graph": (
        24_561,
        "70c1a41f9c826fdf47dc413fe76876b67e2b17abb00daba0e935d75f9dbaaa1e",
    ),
    "imputation_transfer_execution": (
        115,
        "545f626e7b9c6aed25a5ae8dd3156c96db2cd35e5637ec1ebe443432d1684277",
    ),
    "imputation_waiver_records": (
        70,
        "8075fc95d112a7442fa1d9e0f5a6a6d27ad081039147569d635993219eeac92a",
    ),
    "publication_attempts": (
        4,
        "44ce6aff602656a91fce128c7f81e14b80923b78e8ab9736d806c450a95285d6",
    ),
    "publication_audit_chain": (
        1,
        "e7c1984b50696829b96f48cf902b7397931d0444b50deb8ace644b50a07945ac",
    ),
    "publication_promotion": (
        6,
        "4720ab158b1c4ca52dc3276035a819abb7a3b44b6e1216807efc480b9627b6be",
    ),
    "publication_release": (
        19,
        "7e40695a85d765e4ba213cbb640f85a5ce1e53f3db06e298dab80b0299293984",
    ),
    "publication_release_graph": (
        3,
        "6a781915fd491d2c4b16d2b7d482f69cf362c904130093c59f9629f7a319269b",
    ),
    "resolved_seed_protocol": (
        1_032,
        "6d4af2d988ee87ed38a044c1007e854176af17be4782e802d86d960f10376f7a",
    ),
    "resolved_seed_site_bindings": (
        316,
        "199ec98045871070860164b522892327ab840b48c0c6a53972ebc0ff0d7d091f",
    ),
    "resolved_vintage_authorities": (
        63,
        "5f2edfe9826e1200e2b8706780c9f53b7c61f01aad8cab5907f08c37a0b1ead2",
    ),
    "selection": (
        87,
        "a1e8d271197566eb3cf23b309156c2efb563ae10fe21968a3cfe9b9d827da2db",
    ),
    "source_geography_identity": (
        8,
        "364a30f79a717a3627ae5e972c753d98a7727ec30d41682ac5b3f5d8027b6274",
    ),
    "source_pins": (
        74,
        "a6100cd55d2e4f450208b6b0d3513f3d56c5068f74a54dfe62705d9a5f33ef07",
    ),
    "source_stage_asset": (
        2,
        "06bdd99bdd2837c0ca71b096f10f956141b592aff6496cd632cfadec5dfe66d3",
    ),
    "source_stage_manifest": (
        2,
        "d6782c5de5bbed1bdc6bf653c4a6d4aadcad4ccc72d35e1092e130fcb04680a3",
    ),
    "source_stages": (
        1_715,
        "78227f34c3f4ef946c06693dec9177dd906069f98016854bfd4d39c4cd4c51fa",
    ),
    "spine_assembly_household_mass_shares": (
        2,
        "2ef5d8ca4955d8fa99c723192e4cd948adfc5cb6d05b0c1edef1e28d110cdb2f",
    ),
    "spine_assembly_mass_anchor_channel": (
        1,
        "8e7beb4622b10dfd98e93c44d5bde1054e9ffc6f5b7d6875abb1d599531dd890",
    ),
    "spine_assembly_shared_dtype_policy": (
        1,
        "f90636dc22d47983ffa458ec37acb9f18ac27e5da3e7788bfe5f47ac6e81b53f",
    ),
    "spine_channels": (
        7,
        "cf0000464013118955571dac2691d4cc1b900c97c6570349b489678cdf937649",
    ),
    "spine_pipeline_contract": (
        89,
        "8d281f8f1e2f9914684150b0424ff816d0ec09ed0d9053ea5a64e877a86cc35b",
    ),
    "spine_sampling": (
        17,
        "29c6c1b3243e178783e7ab139993ba3a9b42d62edd4e9e4e1f3b28688daf2c6d",
    ),
    "spine_seed_site_bindings": (
        316,
        "bb848f38e068858ac23e964a4ffbbf5542253da50515c72fe7753d8c56912486",
    ),
    "spine_support_roles": (
        29,
        "287f107f82a17a06c57cf50fbd81532ab616f7a1f8c0fdd2038a35065291addf",
    ),
    "spine_support_source_pool": (
        13,
        "6a33aa6dd4c2d446e5fa5fc13a8a79b6d94183f9d343de11e3a3de5e469c19da",
    ),
    "spine_support_source_pool_metadata": (
        2,
        "d0de9c71fb388104cbd2fad9889b1eb18dd0b844c7344fd8a75ffda9d3cb9814",
    ),
    "take_up": (
        328,
        "d53096196db4c34da260cce2f35af8e7ba67f978448656c602ee5e17529dc4e0",
    ),
    "vintage_records": (
        112,
        "c69a923e893fde8a929ca087d29ded607fa9c04fd14a39d0a401ad1db845e6de",
    ),
}


def _claim(
    claim_id: str,
    source_prefix: str,
    mode: UsageMode,
    effect: Generation0Effect,
    consumer: str,
    verifier: str,
    *,
    legacy_sinks: tuple[str, ...] = (),
    relative_sink_prefix: str | None = None,
    rationale: str | None = None,
    pointer_class: str = "all",
) -> UsageClaim:
    expected_count, expected_sha256 = _PINS.get(claim_id, (-1, ""))
    return UsageClaim(
        id=claim_id,
        source_prefix=source_prefix,
        mode=mode,
        generation0_effect=effect,
        consumer=consumer,
        verifier=verifier,
        expected_pointer_count=expected_count,
        expected_pointer_sha256=expected_sha256,
        legacy_sinks=legacy_sinks,
        relative_sink_prefix=relative_sink_prefix,
        rationale=rationale,
        pointer_class=pointer_class,
    )


_A = "/authored"
_R = "/resolved"
_LEGACY = Generation0Effect.LEGACY_BEHAVIOR
_NO_EFFECT = Generation0Effect.NO_GENERATION0_EFFECT


def default_usage_claims() -> tuple[UsageClaim, ...]:
    """Return the closed F0 claim set; no claim is inferred from new fields."""

    claims = [
        _claim(
            "country_manifest",
            f"{_A}/country_package.json",
            UsageMode.FRONT_END_VALIDATION,
            _NO_EFFECT,
            "loader.typed_resource_manifest",
            "manifest",
        ),
        _claim(
            "battery",
            f"{_A}/spec~1battery.yaml",
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.battery_contract",
            "legacy",
            legacy_sinks=("/battery_contract", "/stacked_authority_receipt"),
            relative_sink_prefix="/battery_contract",
        ),
        _claim(
            "bundle_country",
            f"{_A}/spec~1bundle.yaml/country",
            UsageMode.FRONT_END_VALIDATION,
            _NO_EFFECT,
            "loader.country_binding",
            "bundle_country",
        ),
        _claim(
            "bundle_dataset_run",
            f"{_A}/spec~1bundle.yaml/dataset_run",
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.stacked_checkpoint_static_components",
            "bundle_dataset",
            legacy_sinks=("/stacked_checkpoint_static_components",),
        ),
        _claim(
            "bundle_identity_generation",
            f"{_A}/spec~1bundle.yaml/identity_generation",
            UsageMode.IDENTITY_ONLY,
            _NO_EFFECT,
            "spec_binding.mirror_attestation",
            "identity",
            rationale=(
                "F0 must not retro-label generation-0 identities; this authored "
                "generation enters only the mirror-attested spec identity until F1"
            ),
        ),
        _claim(
            "bundle_seed_protocol",
            f"{_A}/spec~1bundle.yaml/seed_protocol",
            UsageMode.COMPILER_SEMANTIC,
            _NO_EFFECT,
            "compiler_ir.seed_stream_map.protocol_id",
            "seed_protocol",
        ),
        _claim(
            "calibration",
            f"{_A}/spec~1calibration.yaml",
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.calibration_contract",
            "legacy",
            legacy_sinks=(
                "/calibration_contract",
                "/calibration_tail_contracts",
            ),
            relative_sink_prefix="/calibration_contract",
        ),
        _claim(
            "catalog_metadata_waivers",
            f"{_A}/spec~1catalogs.yaml/metadata_waivers",
            UsageMode.FRONT_END_VALIDATION,
            _NO_EFFECT,
            "typed_closure.metadata_waiver_validation",
            "catalog_waivers",
        ),
        _claim(
            "catalog_columns",
            f"{_A}/spec~1catalogs.yaml/columns",
            UsageMode.COMPILER_SEMANTIC,
            _NO_EFFECT,
            "compiler_ir.typed_inventory.columns",
            "catalog_columns",
        ),
        _claim(
            "geography_phase",
            f"{_A}/spec~1geography.yaml/phase",
            UsageMode.IDENTITY_ONLY,
            _NO_EFFECT,
            "spec_binding.geography_phase",
            "identity",
            rationale=(
                "the migration phase labels the mirrored contract but does not "
                "change the generation-0 stacked geography executor"
            ),
        ),
        _claim(
            "geography_assignment",
            f"{_A}/spec~1geography.yaml/assignment",
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.stacked_checkpoint_static_components.geography_assignment",
            "legacy",
            legacy_sinks=(
                "/stacked_checkpoint_static_components/geography_assignment",
            ),
        ),
    ]

    imputation_roots = {
        "predictor_blocks": UsageMode.LEGACY_BEHAVIOR,
        "models": UsageMode.LEGACY_BEHAVIOR,
        "transfer_execution": UsageMode.LEGACY_BEHAVIOR,
        "gap_fill_schedule": UsageMode.LEGACY_BEHAVIOR,
        "primary_checkpoint": UsageMode.LEGACY_BEHAVIOR,
        "producer_graph": UsageMode.COMPILER_SEMANTIC,
    }
    for root, mode in imputation_roots.items():
        claims.append(
            _claim(
                f"imputation_{root}",
                f"{_A}/spec~1imputation.yaml/{root}",
                mode,
                _LEGACY,
                (
                    "compiler_ir.producer_graph_and_node_slices"
                    if root == "producer_graph"
                    else "legacy_adapter.imputation"
                ),
                "imputation_graph" if root == "producer_graph" else "legacy",
                legacy_sinks=(
                    "/imputation",
                    "/stacked_authority_receipt",
                    "/stacked_checkpoint_static_components",
                ),
            )
        )

    for root in ("chaining", "concepts", "waiver_records"):
        claims.append(
            _claim(
                f"imputation_{root}",
                f"{_A}/spec~1imputation.yaml/{root}",
                UsageMode.FRONT_END_VALIDATION,
                _NO_EFFECT,
                "resolver.imputation_structure_and_concept_closure",
                "imputation_validation",
            )
        )
    claims.extend(
        [
            _claim(
                "imputation_family_execution",
                f"{_A}/spec~1imputation.yaml/families",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.imputation",
                "legacy",
                legacy_sinks=(
                    "/imputation",
                    "/stacked_authority_receipt",
                    "/stacked_checkpoint_static_components",
                ),
                pointer_class="family_execution",
            ),
            _claim(
                "imputation_family_concept_validation",
                f"{_A}/spec~1imputation.yaml/families",
                UsageMode.FRONT_END_VALIDATION,
                _NO_EFFECT,
                "resolver.imputation_concept_closure",
                "imputation_validation",
                pointer_class="family_concept_validation",
            ),
        ]
    )

    for root in ("attempts", "promotion", "audit_chain", "release_graph"):
        claims.append(
            _claim(
                f"publication_{root}",
                f"{_A}/spec~1publication.yaml/{root}",
                UsageMode.IDENTITY_ONLY,
                _NO_EFFECT,
                f"spec_binding.publication_{root}",
                "identity",
                rationale=(
                    f"publication {root} is compiled and identity-attested in F0; "
                    "the generation-0 tool has no corresponding constants payload"
                ),
            )
        )
    claims.extend(
        [
            _claim(
                "publication_release",
                f"{_A}/spec~1publication.yaml/release",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.publication_release",
                "legacy",
                legacy_sinks=("/publication_release", "/spine_sampling"),
                relative_sink_prefix="/publication_release",
            ),
            _claim(
                "selection",
                f"{_A}/spec~1selection.yaml",
                UsageMode.IDENTITY_ONLY,
                _NO_EFFECT,
                "spec_binding.selection_contract",
                "identity",
                rationale=(
                    "selection is a compiled F1 contract; constants_adapter must "
                    "not change generation-0 selection behavior in F0"
                ),
            ),
            _claim(
                "source_geography_identity",
                f"{_A}/spec~1sources.yaml/sources",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.stacked_checkpoint_static_components.geography_assignment",
                "legacy",
                legacy_sinks=(
                    "/stacked_checkpoint_static_components/geography_assignment",
                ),
                pointer_class="stacked_geography_source_identity",
            ),
            _claim(
                "source_pins",
                f"{_A}/spec~1sources.yaml/sources",
                UsageMode.FRONT_END_VALIDATION,
                _NO_EFFECT,
                "loader.sources_schema_and_resolver.source_registry",
                "source_pins",
                pointer_class="source_validation",
            ),
            _claim(
                "source_stage_asset",
                f"{_A}/spec~1sources.yaml/stage_asset",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.imputation.resource_semantics",
                "legacy",
                legacy_sinks=("/imputation",),
            ),
            _claim(
                "source_stage_manifest",
                f"{_A}/spec~1sources.yaml/stage_manifest",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.source_manifest",
                "legacy",
                legacy_sinks=("/source_manifest",),
            ),
            _claim(
                "source_stages",
                f"{_A}/spec~1sources.yaml/stages",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.source_manifest",
                "legacy",
                legacy_sinks=("/source_manifest", "/imputation"),
                relative_sink_prefix="/source_manifest/stages",
            ),
        ]
    )

    spine_routes = {
        "pipeline_contract": (
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.stacked_checkpoint_static_components",
            "legacy",
            ("/stacked_checkpoint_static_components",),
        ),
        "seed_site_bindings": (
            UsageMode.COMPILER_SEMANTIC,
            _NO_EFFECT,
            "compiler_ir.seed_stream_map.owners",
            "seed_bindings",
            (),
        ),
        "channels": (
            UsageMode.FRONT_END_VALIDATION,
            _NO_EFFECT,
            "loader.spine_schema_and_resolver.channel_source_closure",
            "spine_channels",
            (),
        ),
        "sampling": (
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.spine_sampling",
            "legacy",
            ("/spine_sampling", "/stacked_checkpoint_static_components"),
        ),
        "support_roles": (
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.tail_and_imputation_contracts",
            "legacy",
            (
                "/imputation",
                "/calibration_tail_contracts",
                "/stacked_authority_receipt",
                "/stacked_checkpoint_static_components",
            ),
        ),
        "support_source_pool_metadata": (
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.support_spine",
            "legacy",
            ("/support_spine",),
        ),
        "support_source_pool": (
            UsageMode.LEGACY_BEHAVIOR,
            _LEGACY,
            "legacy_adapter.support_spine",
            "legacy",
            ("/support_spine",),
        ),
    }
    for root, (mode, effect, consumer, verifier, sinks) in spine_routes.items():
        claims.append(
            _claim(
                f"spine_{root}",
                f"{_A}/spec~1spine.yaml/{root}",
                mode,
                effect,
                consumer,
                verifier,
                legacy_sinks=sinks,
            )
        )

    claims.extend(
        [
            _claim(
                "spine_assembly_mass_anchor_channel",
                f"{_A}/spec~1spine.yaml/assembly/mass_anchor_channel",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.spine_assembly_and_imputation",
                "spine_assembly_legacy",
                legacy_sinks=(
                    "/spine_assembly/mass_anchor_channel",
                    "/imputation",
                ),
            ),
            _claim(
                "spine_assembly_household_mass_shares",
                f"{_A}/spec~1spine.yaml/assembly/household_mass_shares",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.spine_assembly.household_mass_shares",
                "spine_assembly_legacy",
                legacy_sinks=("/spine_assembly/household_mass_shares",),
                relative_sink_prefix="/spine_assembly/household_mass_shares",
            ),
            _claim(
                "spine_assembly_shared_dtype_policy",
                f"{_A}/spec~1spine.yaml/assembly/shared_dtype_policy",
                UsageMode.FRONT_END_VALIDATION,
                _NO_EFFECT,
                "loader.spine_schema.shared_dtype_policy_const",
                "spine_assembly_validation",
            ),
        ]
    )

    claims.extend(
        [
            _claim(
                "take_up",
                f"{_A}/spec~1take_up.yaml",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.take_up_contract",
                "legacy",
                legacy_sinks=(
                    "/take_up_contract",
                    "/take_up_contract_identity",
                    "/stacked_checkpoint_static_components",
                ),
            ),
            _claim(
                "vintage_records",
                f"{_A}/spec~1vintages.yaml",
                UsageMode.COMPILER_SEMANTIC,
                _NO_EFFECT,
                "resolver.vintage_authorities",
                "vintages",
            ),
            _claim(
                "generated_authorities",
                f"{_R}/generated_authorities",
                UsageMode.LEGACY_BEHAVIOR,
                _LEGACY,
                "legacy_adapter.engine_abi_and_remaining_input_manifest",
                "authorities",
                legacy_sinks=(
                    "/take_up_contract",
                    "/stacked_checkpoint_static_components",
                ),
            ),
            _claim(
                "resolved_seed_protocol",
                f"{_R}/seed_protocol",
                UsageMode.COMPILER_SEMANTIC,
                _NO_EFFECT,
                "compiler_ir.seed_stream_map",
                "seed_protocol",
            ),
            _claim(
                "resolved_seed_site_bindings",
                f"{_R}/seed_site_bindings",
                UsageMode.COMPILER_SEMANTIC,
                _NO_EFFECT,
                "compiler_ir.seed_stream_map.owners",
                "seed_bindings",
            ),
            _claim(
                "resolved_vintage_authorities",
                f"{_R}/vintage_authorities",
                UsageMode.COMPILER_SEMANTIC,
                _NO_EFFECT,
                "compiler_ir.authorities.vintages",
                "vintages",
            ),
        ]
    )
    return tuple(claims)


@dataclass(frozen=True, slots=True)
class _VerificationContext:
    spec: ResolvedSpec
    compiled: CompiledSpecIR
    sources: Mapping[str, object]
    actual_legacy: Mapping[str, object]
    expected_legacy: Mapping[str, object]


def _domain_kind(claim: UsageClaim) -> str:
    marker = f"{_A}/spec~1"
    if not claim.source_prefix.startswith(marker):
        raise FieldUsageError(f"{claim.id}: claim has no domain resource")
    return claim.source_prefix[len(marker) :].split(".yaml", 1)[0]


def _verify_spec_hash(context: _VerificationContext) -> None:
    recomputed = _spec_hash(context.spec, context.sources)
    if recomputed != context.spec.spec_sha256:
        raise FieldUsageError(
            "resolved spec_sha256 differs from the complete normative envelope"
        )


def _verify_legacy(context: _VerificationContext, claim: UsageClaim) -> None:
    for sink in claim.legacy_sinks:
        present_expected, expected = _pointer_value(context.expected_legacy, sink)
        present_actual, actual = _pointer_value(context.actual_legacy, sink)
        if not present_expected:
            raise FieldUsageError(f"{claim.id}: invalid expected legacy sink {sink}")
        if not present_actual:
            raise FieldUsageError(f"{claim.id}: missing legacy sink {sink}")
        differences = diff_legacy_payloads(expected, actual, path=sink)
        if differences:
            paths = ", ".join(row.path for row in differences[:5])
            raise FieldUsageError(
                f"{claim.id}: legacy sink differs from projector at {paths}"
            )


def _verify_manifest(context: _VerificationContext, claim: UsageClaim) -> None:
    _, manifest = _pointer_value(context.sources, claim.source_prefix)
    if not isinstance(manifest, Mapping):
        raise FieldUsageError("country_manifest: object required")
    expected = [resource.descriptor.to_wire() for resource in context.spec.resources]
    if (
        manifest.get("country") != context.spec.country
        or manifest.get("schema_version") != context.spec.schema_version
        or manifest.get("resources") != expected
    ):
        raise FieldUsageError("country_manifest: typed descriptor projection differs")


def _verify_bundle_country(context: _VerificationContext, claim: UsageClaim) -> None:
    _, value = _pointer_value(context.sources, claim.source_prefix)
    if value != context.spec.country:
        raise FieldUsageError("bundle_country: resolved country differs")


def _verify_bundle_dataset(context: _VerificationContext, claim: UsageClaim) -> None:
    _, value = _pointer_value(context.sources, claim.source_prefix)
    if not isinstance(value, Mapping):
        raise FieldUsageError("bundle_dataset_run: object required")
    static = context.actual_legacy.get("stacked_checkpoint_static_components")
    if not isinstance(static, Mapping) or static.get("period") != value.get(
        "target_period"
    ):
        raise FieldUsageError("bundle_dataset_run: checkpoint period differs")
    _verify_legacy(context, claim)


def _verify_domain_schema(
    context: _VerificationContext,
    *,
    kind: ResourceKind,
    schema_id: str,
    claim_id: str,
) -> Mapping[str, object]:
    document = context.spec.domain(kind).to_wire()
    if not isinstance(document, Mapping):
        raise FieldUsageError(f"{claim_id}: typed domain object required")
    try:
        load_schema_registry().validate(document, schema_id)
    except SpecValidationError as error:
        raise FieldUsageError(
            f"{claim_id}: schema validation differs: {error}"
        ) from error
    return document


def _verify_source_pins(context: _VerificationContext, claim: UsageClaim) -> None:
    document = _verify_domain_schema(
        context,
        kind=ResourceKind.SOURCES,
        schema_id="sources.schema.json",
        claim_id=claim.id,
    )
    rows = document.get("sources")
    if not isinstance(rows, (list, tuple)):
        raise FieldUsageError("source_pins: source registry array required")
    ids = [
        str(row.get("id"))
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    ]
    if len(ids) != len(rows) or len(ids) != len(set(ids)):
        raise FieldUsageError(
            "source_pins: source ids are not an exact unique registry"
        )

    expected_refs: set[tuple[str, str, str]] = set()
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):  # schema validation is the primary guard
            raise FieldUsageError(f"source_pins: source row {index} is not an object")
        loader = value.get("loader")
        if isinstance(loader, str) and loader.startswith("kernel:"):
            expected_refs.add(
                (
                    "kernel",
                    loader.removeprefix("kernel:"),
                    f"sources/sources/{index}/loader",
                )
            )
        vintages = value.get("vintages", ())
        if isinstance(vintages, (list, tuple)):
            for vintage_index, vintage in enumerate(vintages):
                if isinstance(vintage, str) and vintage.startswith("vintage:"):
                    expected_refs.add(
                        (
                            "vintage",
                            vintage.removeprefix("vintage:"),
                            f"sources/sources/{index}/vintages/{vintage_index}",
                        )
                    )
    actual_refs = {
        (reference.namespace, reference.id, reference.source_path)
        for reference in context.spec.references
    }
    missing = sorted(expected_refs - actual_refs)
    if missing:
        raise FieldUsageError(
            f"source_pins: resolver omitted {len(missing)} source reference(s): "
            f"{missing[:3]!r}"
        )


def _verify_spine_channels(context: _VerificationContext, claim: UsageClaim) -> None:
    spine = _verify_domain_schema(
        context,
        kind=ResourceKind.SPINE,
        schema_id="spine.schema.json",
        claim_id=claim.id,
    )
    sources = _verify_domain_schema(
        context,
        kind=ResourceKind.SOURCES,
        schema_id="sources.schema.json",
        claim_id=claim.id,
    )
    source_rows = sources.get("sources")
    channel_rows = spine.get("channels")
    if not isinstance(source_rows, (list, tuple)) or not isinstance(
        channel_rows, (list, tuple)
    ):
        raise FieldUsageError("spine_channels: source and channel arrays required")
    source_ids = {
        str(row["id"])
        for row in source_rows
        if isinstance(row, Mapping) and "id" in row
    }
    channel_ids: list[str] = []
    for index, value in enumerate(channel_rows):
        if not isinstance(value, Mapping):
            raise FieldUsageError(
                f"spine_channels: channel row {index} is not an object"
            )
        channel_ids.append(str(value["id"]))
        raw_sources = value.get("source")
        channel_sources = (
            (raw_sources,) if isinstance(raw_sources, str) else raw_sources
        )
        if not isinstance(channel_sources, (list, tuple)) or any(
            source not in source_ids for source in channel_sources
        ):
            raise FieldUsageError(
                f"spine_channels: channel {value['id']!r} has a dangling source"
            )
    if len(channel_ids) != len(set(channel_ids)):
        raise FieldUsageError("spine_channels: channel ids are not unique")
    assembly = spine.get("assembly")
    if not isinstance(assembly, Mapping) or assembly.get(
        "mass_anchor_channel"
    ) not in set(channel_ids):
        raise FieldUsageError("spine_channels: assembly anchor is not a channel")


def _verify_spine_assembly_legacy(
    context: _VerificationContext,
    claim: UsageClaim,
) -> None:
    _verify_domain_schema(
        context,
        kind=ResourceKind.SPINE,
        schema_id="spine.schema.json",
        claim_id=claim.id,
    )
    _, source = _pointer_value(context.sources, claim.source_prefix)
    suffix = claim.source_prefix.split("/assembly/", 1)[1]
    present, sink = _pointer_value(context.actual_legacy, f"/spine_assembly/{suffix}")
    if not present or not (
        _semantic_subset(source, sink) and _semantic_subset(sink, source)
    ):
        raise FieldUsageError(
            f"{claim.id}: legacy spine_assembly sink differs from its source"
        )
    _verify_legacy(context, claim)


def _verify_spine_assembly_validation(
    context: _VerificationContext,
    claim: UsageClaim,
) -> None:
    _verify_domain_schema(
        context,
        kind=ResourceKind.SPINE,
        schema_id="spine.schema.json",
        claim_id=claim.id,
    )
    _, value = _pointer_value(context.sources, claim.source_prefix)
    if value != "canonical_string_storage":
        raise FieldUsageError(
            "spine_assembly_shared_dtype_policy: unsupported storage policy"
        )


def _verify_compiled_domain(context: _VerificationContext, claim: UsageClaim) -> None:
    kind = ResourceKind(_domain_kind(claim))
    _, source = _pointer_value(context.sources, claim.source_prefix)
    domain = context.spec.domain(kind).to_wire()
    suffix = claim.source_prefix.split(".yaml", 1)[1]
    if suffix:
        present, domain = _pointer_value(domain, suffix)
        if not present:
            raise FieldUsageError(f"{claim.id}: missing typed domain binding")
    if not _semantic_subset(source, domain):
        raise FieldUsageError(f"{claim.id}: typed domain binding differs")


def _verify_catalog_columns(context: _VerificationContext, claim: UsageClaim) -> None:
    _verify_compiled_domain(context, claim)
    inventory = _wire(context.compiled.typed_inventory)
    if not isinstance(inventory, Mapping):
        raise FieldUsageError("catalog_columns: typed inventory object required")
    expected = [
        {
            "key": column.key,
            "entity": column.entity.id,
            "dtype": column.dtype,
            "unit": column.unit,
            "period": column.period,
            "vintage": column.vintage,
            "nullable": column.nullable,
            "domain": column.domain,
            "public_stability": column.public_stability,
            "unit_waiver": column.unit_waiver,
        }
        for column in context.spec.columns
    ]
    if inventory.get("columns") != expected:
        raise FieldUsageError("catalog_columns: typed column projection differs")


def _verify_catalog_waivers(context: _VerificationContext, claim: UsageClaim) -> None:
    _verify_compiled_domain(context, claim)
    _, waivers = _pointer_value(context.sources, claim.source_prefix)
    if not isinstance(waivers, (list, tuple)):
        raise FieldUsageError("catalog_metadata_waivers: array required")
    waiver_ids = {
        str(waiver["id"])
        for waiver in waivers
        if isinstance(waiver, Mapping) and "id" in waiver
    }
    used = {column.unit_waiver for column in context.spec.columns}
    used.discard(None)
    if waiver_ids != used:
        raise FieldUsageError("catalog_metadata_waivers: typed closure differs")


def _verify_imputation_validation(
    context: _VerificationContext,
    claim: UsageClaim,
) -> None:
    _verify_compiled_domain(context, claim)
    document = context.spec.domain(ResourceKind.IMPUTATION).to_wire()
    families = document.get("families", [])
    graph = document.get("producer_graph", {})
    if not isinstance(families, (list, tuple)) or not isinstance(graph, Mapping):
        raise FieldUsageError("imputation validation: typed structures are absent")
    family_ids = frozenset(
        str(row["id"]) for row in families if isinstance(row, Mapping) and "id" in row
    )
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, (list, tuple)):
        raise FieldUsageError("imputation validation: producer nodes are absent")
    node_ids = frozenset(
        str(row["id"]) for row in nodes if isinstance(row, Mapping) and "id" in row
    )
    try:
        _validate_imputation_concept_coverage(document, families=families)
        _validate_imputation_structure(
            document,
            families=families,
            family_ids=family_ids,
            graph=graph,
            node_ids=node_ids,
        )
    except (TypeError, ValueError) as error:
        raise FieldUsageError(
            f"{claim.id}: imputation validation closure differs: {error}"
        ) from error


def _verify_imputation_graph(context: _VerificationContext, claim: UsageClaim) -> None:
    _verify_compiled_domain(context, claim)
    _, source = _pointer_value(context.sources, claim.source_prefix)
    authored = context.compiled.producer_graph.authored
    if authored is None or not _semantic_subset(source, _wire(authored)):
        raise FieldUsageError("imputation_producer_graph: compiled graph differs")
    for node in context.compiled.producer_graph.nodes:
        expected_path = f"/imputation/producer_graph/nodes/{node.id}"
        params = {
            param.path: _wire(param.value)
            for param in context.compiled.nodes[
                context.compiled.producer_graph.order.index(node.id)
            ].resolved_params
        }
        if params.get(expected_path) != _wire(node.source):
            raise FieldUsageError(
                f"imputation_producer_graph: node slice missing {node.id}"
            )
    _verify_legacy(context, claim)


def _verify_seed_protocol(context: _VerificationContext, claim: UsageClaim) -> None:
    protocol = context.spec.seed_protocol
    compiled = context.compiled.seed_stream_map
    if (
        compiled.protocol_id != protocol.id
        or compiled.implementation_id != protocol.implementation_id
        or compiled.implementation_sha256 != protocol.implementation_sha256
    ):
        raise FieldUsageError("seed_protocol: compiler identity differs")
    compiled_sites = {site.id: site.to_wire() for site in compiled.sites}
    expected_sites = {}
    for site in protocol.sites:
        wire = site.to_wire()
        site_id = str(wire.pop("id"))
        stream = wire.pop("stream")
        expected_sites[site_id] = {
            "id": site_id,
            "stream": stream,
            "contract": wire,
            "owners": compiled_sites.get(site_id, {}).get("owners", []),
        }
    if compiled_sites != expected_sites:
        raise FieldUsageError("seed_protocol: site contracts differ")
    declared_streams = set(protocol.streams)
    consumed_streams = {site.stream for site in protocol.sites}
    if declared_streams != consumed_streams:
        raise FieldUsageError("seed_protocol: stream registry is not exact")


def _verify_seed_bindings(context: _VerificationContext, claim: UsageClaim) -> None:
    expected = {
        binding.site: tuple((owner.kind.value, owner.id) for owner in binding.owners)
        for binding in context.spec.seed_site_bindings
    }
    actual = {site.id: site.owners for site in context.compiled.seed_stream_map.sites}
    if expected != actual:
        raise FieldUsageError("seed_site_bindings: compiled site owners differ")
    reverse: dict[tuple[str, str], list[str]] = {}
    for site, owners in expected.items():
        for owner in owners:
            reverse.setdefault(owner, []).append(site)
    owner_rows = {
        (owner.kind, owner.id): list(owner.sites)
        for owner in context.compiled.seed_stream_map.owners
    }
    if reverse != owner_rows:
        raise FieldUsageError("seed_site_bindings: compiled owner rows differ")


def _verify_vintages(context: _VerificationContext, claim: UsageClaim) -> None:
    if context.compiled.vintage_authorities != context.spec.vintage_authorities:
        raise FieldUsageError("vintage_authorities: compiled authority differs")
    if claim.source_prefix.startswith(f"{_A}/"):
        _verify_compiled_domain(context, claim)


def _verify_authorities(context: _VerificationContext, claim: UsageClaim) -> None:
    if context.compiled.generated_authorities != context.spec.generated_authorities:
        raise FieldUsageError("generated_authorities: compiled authority differs")
    generated = _wire(context.spec.generated_authorities)
    if not isinstance(generated, Mapping):
        raise FieldUsageError("generated_authorities: object required")
    lock = generated.get("engine_abi_lock")
    static = context.actual_legacy.get("stacked_checkpoint_static_components")
    if not isinstance(lock, Mapping) or not isinstance(static, Mapping):
        raise FieldUsageError("generated_authorities: engine/static objects required")
    pool_code = static.get("pool_code")
    if not isinstance(pool_code, Mapping):
        raise FieldUsageError("generated_authorities: pool_code object required")
    engine = lock.get("engine")
    if not isinstance(engine, Mapping) or static.get(
        "policyengine_us_version"
    ) != engine.get("version"):
        raise FieldUsageError("generated_authorities: engine version differs")
    manifest = lock.get("remaining_stage_input_manifest")
    if not isinstance(manifest, Mapping) or not isinstance(
        manifest.get("rows"), (list, tuple)
    ):
        raise FieldUsageError("generated_authorities: remaining-stage rows are absent")
    if not isinstance(pool_code.get("remaining_stage_input_manifest"), Mapping):
        raise FieldUsageError(
            "generated_authorities: compiled remaining-stage receipt is absent"
        )
    _verify_legacy(context, claim)


def _verify_claim(context: _VerificationContext, claim: UsageClaim) -> None:
    if "normalized_resources" in claim.consumer or "surfaces" in claim.consumer:
        raise FieldUsageError(
            f"{claim.id}: passthrough storage cannot be semantic evidence"
        )
    if claim.mode is UsageMode.IDENTITY_ONLY and not claim.rationale:
        raise FieldUsageError(f"{claim.id}: identity-only claim requires rationale")
    if claim.mode is not UsageMode.IDENTITY_ONLY and claim.rationale is not None:
        raise FieldUsageError(f"{claim.id}: rationale is reserved for identity-only")
    if claim.mode is UsageMode.LEGACY_BEHAVIOR and (
        claim.generation0_effect is not Generation0Effect.LEGACY_BEHAVIOR
    ):
        raise FieldUsageError(f"{claim.id}: legacy mode must have legacy effect")

    verifiers = {
        "identity": lambda: _verify_spec_hash(context),
        "manifest": lambda: _verify_manifest(context, claim),
        "bundle_country": lambda: _verify_bundle_country(context, claim),
        "bundle_dataset": lambda: _verify_bundle_dataset(context, claim),
        "legacy": lambda: _verify_legacy(context, claim),
        "source_pins": lambda: _verify_source_pins(context, claim),
        "spine_channels": lambda: _verify_spine_channels(context, claim),
        "spine_assembly_legacy": lambda: _verify_spine_assembly_legacy(context, claim),
        "spine_assembly_validation": lambda: _verify_spine_assembly_validation(
            context, claim
        ),
        "catalog_columns": lambda: _verify_catalog_columns(context, claim),
        "catalog_waivers": lambda: _verify_catalog_waivers(context, claim),
        "imputation_validation": lambda: _verify_imputation_validation(context, claim),
        "imputation_graph": lambda: _verify_imputation_graph(context, claim),
        "seed_protocol": lambda: _verify_seed_protocol(context, claim),
        "seed_bindings": lambda: _verify_seed_bindings(context, claim),
        "vintages": lambda: _verify_vintages(context, claim),
        "authorities": lambda: _verify_authorities(context, claim),
    }
    verifier = verifiers.get(claim.verifier)
    if verifier is None:
        raise FieldUsageError(f"{claim.id}: unknown verifier {claim.verifier!r}")
    verifier()


def _sink_pointers(claim: UsageClaim, pointer: str) -> tuple[str, ...]:
    sinks = list(claim.legacy_sinks)
    if claim.relative_sink_prefix is not None:
        suffix = pointer.removeprefix(claim.source_prefix)
        sinks.insert(0, f"{claim.relative_sink_prefix}{suffix}")
    if claim.id == "calibration" and pointer.endswith(
        "/solver/stopping_contract/max_epochs"
    ):
        sinks.append("/calibration_contract/solver/stopping/max_epochs")
    if claim.mode is UsageMode.IDENTITY_ONLY:
        sinks.append("/spec_binding/spec_sha256")
    return tuple(dict.fromkeys(sinks))


def build_field_usage_ledger(
    spec: ResolvedSpec,
    *,
    compiled: CompiledSpecIR | None = None,
    legacy_payload: Mapping[str, object] | None = None,
    claims: Sequence[UsageClaim] | None = None,
    enforce_expected_total: bool = True,
) -> FieldUsageLedger:
    """Build or refuse the complete exact-pointer F0 usage ledger."""

    if not isinstance(spec, ResolvedSpec):
        raise TypeError("build_field_usage_ledger requires a ResolvedSpec")
    if spec.country != "us":
        raise FieldUsageError("the generation-0 field-usage ledger is US-specific")
    compiled = compile_spec(spec) if compiled is None else compiled
    expected_legacy = compile_to_legacy_payload(spec)
    actual_legacy = (
        expected_legacy if legacy_payload is None else deepcopy(dict(legacy_payload))
    )
    sources = configuration_sources(spec)
    _verify_spec_hash(
        _VerificationContext(
            spec=spec,
            compiled=compiled,
            sources=sources,
            actual_legacy=actual_legacy,
            expected_legacy=expected_legacy,
        )
    )
    all_rows = _terminal_rows(sources, path="")
    universe = {path: value for path, value in all_rows}
    if len(universe) != len(all_rows):  # pragma: no cover - pointer construction
        raise FieldUsageError("configuration pointer universe contains duplicates")
    if enforce_expected_total and len(universe) != EXPECTED_CONFIGURATION_FIELD_COUNT:
        raise FieldUsageError(
            "configuration field count changed: "
            f"expected {EXPECTED_CONFIGURATION_FIELD_COUNT}, got {len(universe)}"
        )
    authored_count = sum(path.startswith("/authored/") for path in universe)
    resolved_count = sum(path.startswith("/resolved/") for path in universe)
    if enforce_expected_total and (
        authored_count != EXPECTED_AUTHORED_FIELD_COUNT
        or resolved_count != EXPECTED_RESOLVED_BINDING_FIELD_COUNT
    ):
        raise FieldUsageError(
            "configuration source counts changed: expected "
            f"{EXPECTED_AUTHORED_FIELD_COUNT} authored and "
            f"{EXPECTED_RESOLVED_BINDING_FIELD_COUNT} resolved, got "
            f"{authored_count} and {resolved_count}"
        )

    selected_claims = tuple(default_usage_claims() if claims is None else claims)
    context = _VerificationContext(
        spec=spec,
        compiled=compiled,
        sources=sources,
        actual_legacy=actual_legacy,
        expected_legacy=expected_legacy,
    )
    claimed_by: dict[str, list[UsageClaim]] = {}
    receipts: list[ClaimReceipt] = []
    for claim in selected_claims:
        present, subtree = _pointer_value(sources, claim.source_prefix)
        if not present:
            raise FieldUsageError(
                f"{claim.id}: source prefix matched zero fields: {claim.source_prefix}"
            )
        rows = _claim_rows(claim, subtree)
        if not rows:
            raise FieldUsageError(f"{claim.id}: source prefix matched zero fields")
        count, digest = _path_inventory(rows)
        if (
            count != claim.expected_pointer_count
            or digest != claim.expected_pointer_sha256
        ):
            raise FieldUsageError(
                f"{claim.id}: stale pointer claim; expected "
                f"{claim.expected_pointer_count}/{claim.expected_pointer_sha256}, "
                f"got {count}/{digest}"
            )
        _verify_claim(context, claim)
        receipts.append(ClaimReceipt(claim.id, count, digest))
        for pointer, _ in rows:
            claimed_by.setdefault(pointer, []).append(claim)

    unclaimed = sorted(set(universe) - set(claimed_by))
    if unclaimed:
        sample = ", ".join(unclaimed[:10])
        raise FieldUsageError(
            f"{len(unclaimed)} unclaimed normative field(s): {sample}"
        )
    unknown = sorted(set(claimed_by) - set(universe))
    if unknown:  # pragma: no cover - expansion starts from the universe
        raise FieldUsageError(f"claims produced unknown pointers: {unknown[:10]!r}")
    multiple = sorted(
        pointer for pointer, owners in claimed_by.items() if len(owners) != 1
    )
    if multiple:
        detail = ", ".join(
            f"{pointer}={','.join(claim.id for claim in claimed_by[pointer])}"
            for pointer in multiple[:10]
        )
        raise FieldUsageError(
            f"{len(multiple)} field(s) have multiple primary claims: {detail}"
        )

    fields = tuple(
        FieldUse(
            pointer=pointer,
            value_sha256=sha256_json(universe[pointer]),
            claim_id=claimed_by[pointer][0].id,
            mode=claimed_by[pointer][0].mode,
            generation0_effect=claimed_by[pointer][0].generation0_effect,
            consumer=claimed_by[pointer][0].consumer,
            sink_pointers=_sink_pointers(claimed_by[pointer][0], pointer),
            proof=claimed_by[pointer][0].verifier,
        )
        for pointer in sorted(universe)
    )
    return FieldUsageLedger(fields=fields, claims=tuple(receipts))


def claim_expansion_inventory(spec: ResolvedSpec) -> dict[str, tuple[int, str]]:
    """Developer helper for reviewing a deliberately changed closed claim set."""

    sources = configuration_sources(spec)
    result: dict[str, tuple[int, str]] = {}
    for claim in default_usage_claims():
        present, subtree = _pointer_value(sources, claim.source_prefix)
        if not present:
            result[claim.id] = (0, "")
            continue
        result[claim.id] = _path_inventory(_claim_rows(claim, subtree))
    return result


__all__ = [
    "EXPECTED_AUTHORED_FIELD_COUNT",
    "EXPECTED_CONFIGURATION_FIELD_COUNT",
    "EXPECTED_RESOLVED_BINDING_FIELD_COUNT",
    "ClaimReceipt",
    "FieldUsageError",
    "FieldUsageLedger",
    "FieldUse",
    "Generation0Effect",
    "UsageClaim",
    "UsageMode",
    "build_field_usage_ledger",
    "claim_expansion_inventory",
    "configuration_sources",
    "default_usage_claims",
]
