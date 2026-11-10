"""Every kernel the transport registry registers, built engine-free.

``build_transport_registry`` prepares one rules engine per binding before it
registers its kernels, which needs a RuleSpec tree and an engine build. The
kernels' implementation hashes and the derived binding rule's walks read only
the engines' class, so ``transport_kernels`` replays the registry's
registration calls with a stand-in for the prepared engines: an
``AxiomEngine``, the class the registry builds, created without a program.
test_transport_binding_rule holds the replay to the registry itself.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType

from microcosm.build.graph_atomic_geography import register_atomic_geography_kernels
from microcosm.build.transport.codecs import register_transport_codecs
from microcosm.build.transport.evidence_kernels import register_evidence_kernels
from microcosm.build.transport.gate_bindings import TRANSPORT_GATE_REGISTRY
from microcosm.build.transport.gate_kernels import register_gate_kernel
from microcosm.build.transport.person_location_kernels import (
    register_person_location_kernels,
)
from microcosm.build.transport.registry import register_transport_population_kernels
from microcosm.build.transport.reporting_kernels import register_reporting_kernels
from microcosm.build.transport.takeup_kernels import register_takeup_kernels
from microcosm.build.transport.target_kernels import register_target_kernels
from microcosm.build.transport.terminal_kernels import register_terminal_kernels
from microcosm.calibrate.ordered_kernels import CALIBRATE_ORDERED_ADAM
from microcosm.frame.adapters.axiom import AxiomEngine
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateRulesByRefKernel,
    SimulateSolveZeroKernel,
)
from microcosm.graph import KernelRegistry
from microcosm.graph.codecs import SourceCodecRegistry


@functools.cache
def transport_kernels() -> Mapping[str, object]:
    """Every kernel ``build_transport_registry`` registers, by ref, from its
    registration calls with one stand-in engine."""

    engines = MappingProxyType({"stand-in": object.__new__(AxiomEngine)})
    kernels = KernelRegistry()
    codecs = SourceCodecRegistry()
    register_transport_codecs(codecs)
    register_transport_population_kernels(kernels)
    register_atomic_geography_kernels(kernels)
    register_person_location_kernels(kernels)
    register_target_kernels(kernels)
    kernels.register(CALIBRATE_ORDERED_ADAM)
    register_gate_kernel(kernels, TRANSPORT_GATE_REGISTRY)
    register_terminal_kernels(kernels, codecs=codecs)
    kernels.register(SimulateRulesByRefKernel(engines))
    kernels.register(SimulateCounterfactualKernel(engines))
    kernels.register(SimulateSolveZeroKernel(engines))
    register_takeup_kernels(kernels)
    register_reporting_kernels(kernels)
    register_evidence_kernels(kernels, engines)
    return MappingProxyType({ref: kernels.get(ref) for ref in kernels.refs()})
