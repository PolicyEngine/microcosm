"""Bundle-authoritative dynamic stacked-checkpoint identity gates."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from microcosm.build.spec_engine import (
    CompiledSpecIR,
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.model import FrozenMap, freeze_json, thaw_json
from microcosm.build.spec_engine.stacked_authority_semantics import (
    project_stacked_checkpoint_base_identity,
)
from microcosm.build.us_runtime.checkpoint_authority import (
    CheckpointAuthorityError,
)
from microcosm.build.us_runtime.checkpoint_authority import (
    materialize_stacked_checkpoint_base_identity as _materialize_from_plan,
)
from microcosm.build.us_runtime.pool_runtime_plan import (
    USPoolRuntimePlan,
)
from microcosm.build.us_runtime.spec_authority import (
    USSpecAuthority,
    _capability_sha256,
    compile_us_spec_authority,
)


@pytest.fixture(scope="module")
def compiled_us() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


@pytest.fixture(scope="module")
def authority(compiled_us: CompiledSpecIR):
    return compile_us_spec_authority(compile_runtime_authorities(compiled_us))


def materialize_stacked_checkpoint_base_identity(
    authority: USSpecAuthority,
    *,
    input_pins: dict[str, dict[str, object]],
    stack_receipt: dict[str, object],
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object]:
    return _materialize_from_plan(
        USPoolRuntimePlan.from_spec_authority(authority),
        input_pins=input_pins,
        stack_receipt=stack_receipt,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        clone_attachment_fraction=clone_attachment_fraction,
        clone_attachment_seed=clone_attachment_seed,
    )


def _reseal_projections(
    authority: USSpecAuthority,
    projections: FrozenMap,
) -> USSpecAuthority:
    seal = _capability_sha256(
        authority_sha256=authority.authority_sha256,
        spec_sha256=authority.spec_sha256,
        identity_generation=authority.identity_generation,
        behavior=authority._behavior,
        projections=projections,
        declared_sources=authority._declared_sources,
        generated_authorities=authority._generated_authorities,
        vintage_authorities=authority._vintage_authorities,
        execution_abi=authority._execution_abi,
        seed_stream_map=authority._seed_stream_map,
        nodes=authority._nodes,
    )
    return replace(authority, _projections=projections, _seal_sha256=seal)


@pytest.mark.parametrize(
    ("sample_fraction", "sample_seed", "attachment_fraction", "attachment_seed"),
    [
        (0.01, 578, 0.4, 991),
        (0.25, 17, 0.17, 123456),
        (1.0, 0, 1.0, 0),
    ],
)
def test_dynamic_identity_matches_direct_compiler_projection(
    authority,
    compiled_us: CompiledSpecIR,
    sample_fraction: float,
    sample_seed: int,
    attachment_fraction: float,
    attachment_seed: int,
) -> None:
    pins = {
        "zeta": {"sha256": "b" * 64, "size_bytes": 23},
        "alpha": {"sha256": "a" * 64, "size_bytes": 17},
    }
    stack = {
        "sample_fraction": sample_fraction,
        "sample_seed": sample_seed,
        "survey_samples": {"asec": {"rows": 3}, "acs": {"rows": 5}},
        "unicode_probe": "Caf\u00e9",
    }

    actual = materialize_stacked_checkpoint_base_identity(
        authority,
        input_pins=pins,
        stack_receipt=stack,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        clone_attachment_fraction=attachment_fraction,
        clone_attachment_seed=attachment_seed,
    )
    expected = project_stacked_checkpoint_base_identity(
        compiled_us,
        input_pins=pins,
        stack_receipt=stack,
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        clone_attachment_fraction=attachment_fraction,
        clone_attachment_seed=attachment_seed,
    )

    assert actual == expected
    assert list(actual["inputs"]) == ["alpha", "zeta"]


def test_clone_request_updates_nested_resource_identity(authority) -> None:
    stack = {"sample_fraction": 0.01, "sample_seed": 578}
    first = materialize_stacked_checkpoint_base_identity(
        authority,
        input_pins={},
        stack_receipt=stack,
        sample_fraction=0.01,
        sample_seed=578,
        clone_attachment_fraction=0.4,
        clone_attachment_seed=991,
    )
    second = materialize_stacked_checkpoint_base_identity(
        authority,
        input_pins={},
        stack_receipt=stack,
        sample_fraction=0.01,
        sample_seed=578,
        clone_attachment_fraction=0.2,
        clone_attachment_seed=992,
    )
    first_resources = first["pool_code"]["late_producer_resource_semantics"]
    second_resources = second["pool_code"]["late_producer_resource_semantics"]
    assert first_resources != second_resources
    assert first_resources["sha256"] != second_resources["sha256"]


def test_checkpoint_uses_its_dedicated_static_projection(authority) -> None:
    stack = {"sample_fraction": 0.01, "sample_seed": 578}
    kwargs = {
        "input_pins": {},
        "stack_receipt": stack,
        "sample_fraction": 0.01,
        "sample_seed": 578,
        "clone_attachment_fraction": 0.4,
        "clone_attachment_seed": 991,
    }
    baseline = materialize_stacked_checkpoint_base_identity(authority, **kwargs)
    projection_wire = thaw_json(authority._projections)
    assert isinstance(projection_wire, dict)
    imputation = projection_wire["imputation"]
    assert isinstance(imputation, dict)
    imputation["late_producer_resource_semantics"] = {"drift": True}
    projections = freeze_json(projection_wire)
    assert isinstance(projections, FrozenMap)
    drifted = _reseal_projections(authority, projections)

    assert materialize_stacked_checkpoint_base_identity(drifted, **kwargs) == baseline


def test_materializer_is_pure_and_refuses_invalid_requests(authority) -> None:
    stack = {"sample_fraction": 0.01, "sample_seed": 578}
    original = copy.deepcopy(stack)
    materialize_stacked_checkpoint_base_identity(
        authority,
        input_pins={},
        stack_receipt=stack,
        sample_fraction=0.01,
        sample_seed=578,
        clone_attachment_fraction=0.4,
        clone_attachment_seed=991,
    )
    assert stack == original

    with pytest.raises(CheckpointAuthorityError, match="differs"):
        materialize_stacked_checkpoint_base_identity(
            authority,
            input_pins={},
            stack_receipt=stack,
            sample_fraction=0.25,
            sample_seed=578,
            clone_attachment_fraction=0.4,
            clone_attachment_seed=991,
        )
    with pytest.raises(CheckpointAuthorityError, match="lowercase SHA-256"):
        materialize_stacked_checkpoint_base_identity(
            authority,
            input_pins={"source": {"sha256": "invalid", "size_bytes": 1}},
            stack_receipt=stack,
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=0.4,
            clone_attachment_seed=991,
        )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1, float("nan")])
def test_materializer_refuses_invalid_clone_fractions(
    authority,
    fraction: float,
) -> None:
    with pytest.raises(CheckpointAuthorityError, match="clone_attachment_fraction"):
        materialize_stacked_checkpoint_base_identity(
            authority,
            input_pins={},
            stack_receipt={"sample_fraction": 0.01, "sample_seed": 578},
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=fraction,
            clone_attachment_seed=991,
        )


@pytest.mark.parametrize(
    ("field", "request_value", "stack_value"),
    [
        ("sample_seed", 1, True),
        ("sample_fraction", 1.0, True),
    ],
)
def test_materializer_refuses_boolean_stack_request_aliases(
    authority,
    field: str,
    request_value: int | float,
    stack_value: bool,
) -> None:
    stack = {"sample_fraction": 0.01, "sample_seed": 578}
    stack[field] = stack_value
    kwargs = {"sample_fraction": 0.01, "sample_seed": 578}
    kwargs[field] = request_value
    with pytest.raises(CheckpointAuthorityError, match=f"stack_receipt/{field}"):
        materialize_stacked_checkpoint_base_identity(
            authority,
            input_pins={},
            stack_receipt=stack,
            **kwargs,
            clone_attachment_fraction=0.4,
            clone_attachment_seed=991,
        )


@pytest.mark.parametrize("seed", [-1, 2**63, True])
def test_materializer_refuses_out_of_range_seeds(authority, seed: object) -> None:
    with pytest.raises(CheckpointAuthorityError, match="signed 64-bit"):
        materialize_stacked_checkpoint_base_identity(
            authority,
            input_pins={},
            stack_receipt={"sample_fraction": 0.01, "sample_seed": seed},
            sample_fraction=0.01,
            sample_seed=seed,  # type: ignore[arg-type]
            clone_attachment_fraction=0.4,
            clone_attachment_seed=991,
        )


def test_materializer_refuses_non_json_stack_receipt(authority) -> None:
    with pytest.raises(CheckpointAuthorityError, match="canonical JSON"):
        materialize_stacked_checkpoint_base_identity(
            authority,
            input_pins={},
            stack_receipt={
                "sample_fraction": 0.01,
                "sample_seed": 578,
                "invalid": object(),
            },
            sample_fraction=0.01,
            sample_seed=578,
            clone_attachment_fraction=0.4,
            clone_attachment_seed=991,
        )
