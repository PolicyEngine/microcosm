"""Explicit registration for the population side of a transport graph."""

from microcosm.graph import KernelRegistry

from .column_kernels import register_column_kernels
from .geography_kernels import register_geography_kernels
from .population_kernels import register_population_kernels

__all__ = ["register_transport_population_kernels"]


def register_transport_population_kernels(registry: KernelRegistry) -> KernelRegistry:
    """Install the ten population kernels without import-time side effects.

    Source codecs are separately installed with ``register_transport_codecs``;
    target, solver and terminal kernels keep their own registration functions.
    Repeating registration with the same kernel objects is idempotent.
    """
    register_population_kernels(registry)
    register_column_kernels(registry)
    register_geography_kernels(registry)
    return registry
