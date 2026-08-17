"""Pure compiler-to-constants compatibility payload and exact diff support.

F0 ends at this adapter.  It reconstructs the JSON-shaped objects consumed by
the constants-era executor, but never imports that executor or runs a stage.
Tests may compare the result with live generation-0 constructors as an oracle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass

from .battery_semantics import project_battery_legacy_contract
from .calibration_semantics import (
    project_legacy_calibration_contract,
    resolve_calibration_tail_contracts,
)
from .compiler_ir import CompiledSpecIR, compile_spec
from .errors import SpecValidationError
from .imputation_semantics import project_imputation_legacy_payloads
from .model import ResolvedSpec, thaw_json
from .publication_semantics import (
    project_publication_legacy_release,
    project_spine_legacy_sampling,
)
from .stacked_authority_semantics import (
    project_stacked_authority_receipt,
    project_stacked_checkpoint_static_components,
)
from .take_up_semantics import (
    project_legacy_take_up_contract,
    project_legacy_take_up_identity,
)


@dataclass(frozen=True, slots=True)
class LegacyFieldDiff:
    """One exact mismatch between two canonical JSON-shaped payloads."""

    path: str
    reason: str
    expected: object
    actual: object

    def describe(self) -> str:
        return (
            f"{self.path}: {self.reason}; expected={self.expected!r}; "
            f"actual={self.actual!r}"
        )


class LegacyPayloadMismatchError(AssertionError):
    """The compiled adapter differs from the constants-era oracle."""

    def __init__(self, differences: Sequence[LegacyFieldDiff]) -> None:
        self.differences = tuple(differences)
        details = "\n".join(f"- {row.describe()}" for row in self.differences)
        super().__init__(
            f"legacy payload differs at {len(self.differences)} field(s):\n{details}"
        )


def _pointer(parent: str, token: object) -> str:
    escaped = str(token).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped}" if parent else f"/{escaped}"


def diff_legacy_payloads(
    expected: object,
    actual: object,
    *,
    path: str = "",
) -> tuple[LegacyFieldDiff, ...]:
    """Return every field-level difference in deterministic pointer order."""

    differences: list[LegacyFieldDiff] = []
    location = path or "/"
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return (
                LegacyFieldDiff(location, "type differs", "object", type(actual).__name__),
            )
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys, key=str):
            differences.append(
                LegacyFieldDiff(
                    _pointer(path, key),
                    "field missing from actual payload",
                    expected[key],
                    "<missing>",
                )
            )
        for key in sorted(actual_keys - expected_keys, key=str):
            differences.append(
                LegacyFieldDiff(
                    _pointer(path, key),
                    "unexpected field in actual payload",
                    "<missing>",
                    actual[key],
                )
            )
        for key in sorted(expected_keys & actual_keys, key=str):
            differences.extend(
                diff_legacy_payloads(
                    expected[key],
                    actual[key],
                    path=_pointer(path, key),
                )
            )
        return tuple(differences)

    if isinstance(expected, list | tuple):
        if not isinstance(actual, list | tuple):
            return (
                LegacyFieldDiff(location, "type differs", "array", type(actual).__name__),
            )
        common = min(len(expected), len(actual))
        for index in range(common):
            differences.extend(
                diff_legacy_payloads(
                    expected[index],
                    actual[index],
                    path=_pointer(path, index),
                )
            )
        for index in range(common, len(expected)):
            differences.append(
                LegacyFieldDiff(
                    _pointer(path, index),
                    "array item missing from actual payload",
                    expected[index],
                    "<missing>",
                )
            )
        for index in range(common, len(actual)):
            differences.append(
                LegacyFieldDiff(
                    _pointer(path, index),
                    "unexpected array item in actual payload",
                    "<missing>",
                    actual[index],
                )
            )
        return tuple(differences)

    if type(expected) is not type(actual):
        return (
            LegacyFieldDiff(
                location,
                "scalar type differs",
                expected,
                actual,
            ),
        )
    if expected != actual:
        return (LegacyFieldDiff(location, "value differs", expected, actual),)
    return ()


def assert_legacy_payload_equal(expected: object, actual: object) -> None:
    """Raise one field-complete diagnostic when canonical payloads differ."""

    differences = diff_legacy_payloads(expected, actual)
    if differences:
        raise LegacyPayloadMismatchError(differences)


def _document(ir: CompiledSpecIR, kind: str) -> dict[str, object]:
    document = ir.resource(kind)
    return deepcopy(dict(document))


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _source_manifest(sources: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(
        sources.get("stage_manifest"), location="sources/stage_manifest"
    )
    stages = sources.get("stages")
    if not isinstance(stages, list):
        raise SpecValidationError("sources/stages: array required")
    projected_stages = deepcopy(stages)
    # Canonical normalization intentionally represents integral JSON numbers
    # as integers.  The WIC rate ABI predates that rule and its rate vector is
    # a floating-point runtime contract, including its zero-probability row.
    # Restore that one semantic type at the compiler boundary.
    for stage_index, stage_value in enumerate(projected_stages):
        stage = _mapping(stage_value, location=f"sources/stages/{stage_index}")
        operations = stage.get("operations")
        if not isinstance(operations, list):
            raise SpecValidationError(
                f"sources/stages/{stage_index}/operations: array required"
            )
        for operation_index, operation_value in enumerate(operations):
            operation = _mapping(
                operation_value,
                location=(
                    f"sources/stages/{stage_index}/operations/{operation_index}"
                ),
            )
            if operation.get("kind") != "derive_wic_claim":
                continue
            rates = _mapping(
                operation.get("category_rates"),
                location=(
                    f"sources/stages/{stage_index}/operations/{operation_index}/"
                    "category_rates"
                ),
            )
            values = _mapping(
                rates.get("values"),
                location=(
                    f"sources/stages/{stage_index}/operations/{operation_index}/"
                    "category_rates/values"
                ),
            )
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise SpecValidationError(
                        "derive_wic_claim category rates must be numeric"
                    )
                values[key] = float(value)  # type: ignore[index]
    return {**deepcopy(dict(metadata)), "stages": projected_stages}


def _support_spine(spine: Mapping[str, object]) -> dict[str, object]:
    metadata = _mapping(
        spine.get("support_source_pool_metadata"),
        location="spine/support_source_pool_metadata",
    )
    support = _mapping(
        spine.get("support_source_pool"),
        location="spine/support_source_pool",
    )
    return {
        **deepcopy(dict(metadata)),
        "support_spine": deepcopy(dict(support)),
    }


def _engine_abi_lock(spec: ResolvedSpec) -> dict[str, object]:
    generated = thaw_json(spec.generated_authorities)
    if not isinstance(generated, dict):  # pragma: no cover - model invariant
        raise SpecValidationError("generated authorities: object required")
    return deepcopy(
        dict(
            _mapping(
                generated.get("engine_abi_lock"),
                location="generated authorities/engine_abi_lock",
            )
        )
    )


def compile_to_legacy_payload(spec: ResolvedSpec) -> dict[str, object]:
    """Compile the US bundle into the constants-era compatibility objects.

    The returned object is data only.  Constructing it neither imports nor
    invokes the generation-0 executor.  Each key names one complete legacy
    consumer surface; exact equality with the live constructors is a required
    F0 gate.  The spec binding deliberately remains ``mirror-attested`` until
    F1 makes the compiled plan the execution authority.
    """

    if not isinstance(spec, ResolvedSpec):
        raise TypeError("compile_to_legacy_payload requires a ResolvedSpec")
    if spec.country != "us":
        raise SpecValidationError(
            "the constants adapter is defined only for the US generation-0 executor"
        )

    ir = compile_spec(spec)
    bundle = _document(ir, "bundle")
    sources = _document(ir, "sources")
    spine = _document(ir, "spine")
    imputation = _document(ir, "imputation")
    take_up = _document(ir, "take_up")
    battery = _document(ir, "battery")
    calibration = _document(ir, "calibration")
    publication = _document(ir, "publication")
    engine_abi_lock = _engine_abi_lock(spec)

    imputation_payload = project_imputation_legacy_payloads(
        imputation,
        sources_document=sources,
        spine_document=spine,
        bundle_document=bundle,
    )
    take_up_contract = project_legacy_take_up_contract(
        take_up,
        engine_abi_lock=engine_abi_lock,
        sources_document=sources,
    )
    authority_receipt = project_stacked_authority_receipt(spec)

    return {
        "source_manifest": _source_manifest(sources),
        "support_spine": _support_spine(spine),
        "imputation": imputation_payload,
        "take_up_contract": take_up_contract,
        "take_up_contract_identity": project_legacy_take_up_identity(
            take_up_contract
        ),
        "calibration_contract": project_legacy_calibration_contract(calibration),
        "calibration_tail_contracts": resolve_calibration_tail_contracts(
            calibration,
            spine_document=spine,
            imputation_document=imputation,
        ),
        "battery_contract": project_battery_legacy_contract(
            battery,
            authority_receipt=authority_receipt,
        ),
        "publication_release": project_publication_legacy_release(publication),
        "spine_sampling": project_spine_legacy_sampling(
            spine,
            publication=publication,
        ),
        "stacked_authority_receipt": authority_receipt,
        "stacked_checkpoint_static_components": (
            project_stacked_checkpoint_static_components(spec)
        ),
    }


__all__ = [
    "LegacyFieldDiff",
    "LegacyPayloadMismatchError",
    "assert_legacy_payload_equal",
    "compile_to_legacy_payload",
    "diff_legacy_payloads",
]
