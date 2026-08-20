"""Bundle-only materialization of the stacked pool's kernel authorities."""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

from microcosm.build.spec_engine import (
    compile_runtime_authorities,
    compile_spec,
    load_bundle,
)
from microcosm.build.spec_engine.model import FrozenMap
from microcosm.build.us_runtime import pool_kernel_authority as kernel_module
from microcosm.build.us_runtime.pool_kernel_authority import (
    USPoolKernelAuthorities,
    USPoolKernelAuthorityError,
)
from microcosm.build.us_runtime.pool_physical_authority import (
    USPoolPhysicalAuthority,
)
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.spec_authority import compile_us_spec_authority
from microcosm.build.us_runtime.stacked_spine import (
    StackedAssemblyAuthority,
    StackedGapFillAuthority,
    StackedLateProducerAuthority,
    StackedPrimaryQrfAuthority,
    StackedTerminalAuthority,
)


@pytest.fixture(scope="module")
def plan() -> USPoolRuntimePlan:
    return USPoolRuntimePlan.from_spec_authority(
        compile_us_spec_authority(
            compile_runtime_authorities(compile_spec(load_bundle("us")))
        )
    )


@pytest.fixture(scope="module")
def authorities(plan: USPoolRuntimePlan) -> USPoolKernelAuthorities:
    return USPoolKernelAuthorities.from_runtime_plan(plan)


def test_materializes_exact_kernel_types_and_preserves_seals(
    plan: USPoolRuntimePlan,
    authorities: USPoolKernelAuthorities,
) -> None:
    assert type(authorities.physical) is USPoolPhysicalAuthority
    assert type(authorities.assembly) is StackedAssemblyAuthority
    assert type(authorities.gap_fill) is StackedGapFillAuthority
    assert type(authorities.primary_qrf) is StackedPrimaryQrfAuthority
    assert type(authorities.late_producers) is StackedLateProducerAuthority
    assert type(authorities.terminal) is StackedTerminalAuthority
    assert authorities.authority_sha256 == plan.authority_sha256
    assert authorities.spec_sha256 == plan.spec_sha256
    assert authorities.physical.authority_sha256 == plan.authority_sha256
    assert authorities.physical.spec_sha256 == plan.spec_sha256


def test_compiles_physical_authority_exactly_once(
    plan: USPoolRuntimePlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    compile_physical = kernel_module.compile_us_pool_physical_authority

    def counted(candidate: USPoolRuntimePlan) -> USPoolPhysicalAuthority:
        nonlocal calls
        calls += 1
        return compile_physical(candidate)

    monkeypatch.setattr(
        kernel_module,
        "compile_us_pool_physical_authority",
        counted,
    )
    materialized = USPoolKernelAuthorities.from_runtime_plan(plan)

    assert calls == 1
    assert materialized.physical.authority_sha256 == plan.authority_sha256


def test_rejects_non_plan_before_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(_candidate: object) -> USPoolPhysicalAuthority:
        raise AssertionError("non-plan value reached the physical compiler")

    monkeypatch.setattr(
        kernel_module,
        "compile_us_pool_physical_authority",
        forbidden,
    )
    with pytest.raises(TypeError, match="USPoolRuntimePlan"):
        USPoolKernelAuthorities.from_runtime_plan(FrozenMap())  # type: ignore[arg-type]


def test_rejects_physical_authority_with_different_plan_seal(
    plan: USPoolRuntimePlan,
    authorities: USPoolKernelAuthorities,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = replace(authorities.physical, spec_sha256="0" * 64)
    monkeypatch.setattr(
        kernel_module,
        "compile_us_pool_physical_authority",
        lambda _candidate: mismatched,
    )

    with pytest.raises(USPoolKernelAuthorityError, match="spec_sha256"):
        USPoolKernelAuthorities.from_runtime_plan(plan)


def test_aggregate_is_frozen(authorities: USPoolKernelAuthorities) -> None:
    with pytest.raises(FrozenInstanceError):
        authorities.spec_sha256 = "0" * 64  # type: ignore[misc]


def test_module_has_no_constants_imports_or_program_named_accessors(
    authorities: USPoolKernelAuthorities,
) -> None:
    source = inspect.getsource(kernel_module)
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert all("constants" not in name.split(".") for name in imported_modules)

    public_names = set(dir(USPoolKernelAuthorities))
    program_ids = {program.id for program in authorities.physical.take_up.programs}
    assert program_ids.isdisjoint(public_names)
    assert all(program_id not in source for program_id in program_ids)
