"""Materialize bundle-issued authorities accepted by stacked pool kernels.

The runtime plan is the sole configuration input.  Its physical authority is
compiled once, retained for the typed leaves consumed outside the legacy
kernel adapters, and projected into each adapter without reopening constants
or the source bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

from microcosm.build.spec_engine.compiler_ir import SeedStreamMap
from microcosm.build.us_runtime.pool_physical_authority import (
    USPoolPhysicalAuthority,
    compile_us_pool_physical_authority,
)
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.stacked_spine import (
    StackedAssemblyAuthority,
    StackedGapFillAuthority,
    StackedLateProducerAuthority,
    StackedPrimaryQrfAuthority,
    StackedTerminalAuthority,
    materialize_stacked_terminal_authority,
)


class USPoolKernelAuthorityError(ValueError):
    """A physical authority does not match the sealed runtime plan."""


@dataclass(frozen=True, slots=True)
class USPoolKernelAuthorities:
    """One immutable, compiler-issued authority set for physical pool kernels."""

    authority_sha256: str
    spec_sha256: str
    seed_stream_map: SeedStreamMap
    physical: USPoolPhysicalAuthority
    assembly: StackedAssemblyAuthority
    gap_fill: StackedGapFillAuthority
    primary_qrf: StackedPrimaryQrfAuthority
    late_producers: StackedLateProducerAuthority
    terminal: StackedTerminalAuthority

    def __post_init__(self) -> None:
        expected_types = (
            ("seed_stream_map", self.seed_stream_map, SeedStreamMap),
            ("physical", self.physical, USPoolPhysicalAuthority),
            ("assembly", self.assembly, StackedAssemblyAuthority),
            ("gap_fill", self.gap_fill, StackedGapFillAuthority),
            ("primary_qrf", self.primary_qrf, StackedPrimaryQrfAuthority),
            ("late_producers", self.late_producers, StackedLateProducerAuthority),
            ("terminal", self.terminal, StackedTerminalAuthority),
        )
        for name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"USPoolKernelAuthorities.{name} requires {expected_type.__name__}"
                )
        for name in ("authority_sha256", "spec_sha256"):
            if getattr(self.physical, name) != getattr(self, name):
                raise USPoolKernelAuthorityError(
                    f"kernel and physical {name} values differ"
                )
        if (
            self.seed_stream_map.implementation_sha256
            != self.physical.seeds.implementation_sha256
        ):
            raise USPoolKernelAuthorityError(
                "kernel seed stream map differs from physical seed authority"
            )

    @classmethod
    def from_runtime_plan(
        cls,
        plan: USPoolRuntimePlan,
    ) -> USPoolKernelAuthorities:
        """Compile ``plan`` once and materialize all accepted kernel adapters."""

        if not isinstance(plan, USPoolRuntimePlan):
            raise TypeError(
                "USPoolKernelAuthorities.from_runtime_plan requires USPoolRuntimePlan"
            )
        physical = compile_us_pool_physical_authority(plan)
        if not isinstance(physical, USPoolPhysicalAuthority):
            raise TypeError(
                "compile_us_pool_physical_authority must return USPoolPhysicalAuthority"
            )
        for name in ("authority_sha256", "spec_sha256"):
            if getattr(physical, name) != getattr(plan, name):
                raise USPoolKernelAuthorityError(
                    f"runtime plan and physical {name} values differ"
                )
        return cls(
            authority_sha256=physical.authority_sha256,
            spec_sha256=physical.spec_sha256,
            seed_stream_map=plan.seed_stream_map,
            physical=physical,
            assembly=StackedAssemblyAuthority(
                **physical.assembly.materializer_kwargs()
            ),
            gap_fill=StackedGapFillAuthority(**physical.gap_fill.materializer_kwargs()),
            primary_qrf=StackedPrimaryQrfAuthority(
                **physical.primary_qrf.materializer_kwargs()
            ),
            late_producers=StackedLateProducerAuthority(
                **physical.late_producers.materializer_kwargs()
            ),
            terminal=materialize_stacked_terminal_authority(
                **physical.terminal.materializer_kwargs()
            ),
        )


__all__ = [
    "USPoolKernelAuthorities",
    "USPoolKernelAuthorityError",
]
