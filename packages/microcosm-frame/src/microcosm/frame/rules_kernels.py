"""Graph kernel routing each node to one of several bound rules engines.

``simulate.rules@1`` (:class:`~microcosm.frame.kernels.SimulateRulesKernel`)
binds one adapter per kernel instance and refuses any other ``engine_ref``,
and a :class:`~microcosm.graph.KernelRegistry` holds one kernel per ref. A run
can therefore reach only one rules engine through ``simulate.rules@1``. An
Axiom adapter wraps one RuleSpec module, so a graph that runs several modules
needs ``simulate.rules_by_ref@1``: it holds an ``engine_ref -> adapter``
mapping and runs each node exactly as ``simulate.rules@1`` does, through the
adapter the node's ``params["engine_ref"]`` names.

Node identity is unchanged by the routing: the node's ``engine_ref`` parameter
already enters its key, and the kernel's implementation hash binds code, not
the set of bound references, so adding a binding of an adapter class already
bound re-keys no existing node.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType

import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.rules as frame_rules_module
import microcosm.frame.schema as frame_schema_module
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.frame.rules import RulesEngine
from microcosm.graph import KernelBase, KernelContext, KernelResult, source_hash

__all__ = ["SimulateRulesByRefKernel"]

_PARAMS = frozenset({"engine_ref", "period", "variables"})


class SimulateRulesByRefKernel(KernelBase):
    """Materialize declared variables through the adapter a node's ref names.

    Args:
        engines: Non-empty mapping from ``engine_ref`` to the adapter it names.
            Each reference must uniquely identify its adapter and that
            adapter's configuration (for an Axiom adapter, use
            :func:`~microcosm.frame.adapters.axiom.axiom_engine_ref`).
        dependencies: Distribution names whose versions affect every bound
            engine's behavior, as for ``simulate.rules@1``.

    Each binding is a :class:`~microcosm.frame.kernels.SimulateRulesKernel`
    held by composition, so a node gets that kernel's parameter contract
    (exactly ``engine_ref``, ``period`` and ``variables``), its declared-
    outputs check, its all-rows rule, and its receipt, unchanged.

    The implementation hash binds this module, ``microcosm.frame.kernels``,
    the frame modules ``simulate.rules@1`` binds, and the source of each
    distinct adapter *class*. It does not bind the references: a second
    binding of a class already bound leaves every node key where it was. A
    binding of a new adapter class adds that class's source to the hash and so
    re-keys every node of this kernel, as replacing an adapter class does for
    ``simulate.rules@1``.

    Raises:
        TypeError: If ``engines`` is not a mapping, or a binding is not a
            :class:`~microcosm.frame.rules.RulesEngine`.
        ValueError: If ``engines`` is empty, or a reference is empty.
    """

    ref = "simulate.rules_by_ref@1"

    def __init__(
        self,
        engines: Mapping[str, RulesEngine],
        dependencies: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(engines, Mapping):
            raise TypeError(
                "engines must be a mapping from engine_ref to a RulesEngine."
            )
        if not engines:
            raise ValueError("engines must bind at least one engine_ref.")
        delegates = {
            engine_ref: SimulateRulesKernel(engine_ref, engine, dependencies)
            for engine_ref, engine in engines.items()
        }
        self._delegates = MappingProxyType(dict(sorted(delegates.items())))
        self._adapter_classes = tuple(
            sorted(
                {type(engine) for engine in engines.values()},
                key=lambda cls: (cls.__module__, cls.__qualname__),
            )
        )
        # Every delegate declares the same contract; take it from one so the
        # two kernels cannot drift apart.
        self.capabilities = next(iter(self._delegates.values())).capabilities

    def engine_refs(self) -> tuple[str, ...]:
        """The bound references, sorted."""

        return tuple(self._delegates)

    def implementation_hash(self) -> str:
        """Hash the routing and delegate kernels plus every bound adapter class."""

        return source_hash(
            type(self),
            SimulateRulesKernel,
            *self._adapter_classes,
            frame_bundle_module,
            frame_rules_module,
            frame_schema_module,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        """Run the node through the delegate its ``engine_ref`` names.

        The parameter set and the reference's type are checked before
        routing, in the order and with the exception types
        ``simulate.rules@1`` uses, so a malformed node fails the same way
        under either kernel.
        """

        actual = set(context.params)
        if actual != _PARAMS:
            raise ValueError(
                "simulate.rules_by_ref parameters must be exactly "
                f"{sorted(_PARAMS)!r}; got {sorted(actual)!r}."
            )
        engine_ref = context.params["engine_ref"]
        if not isinstance(engine_ref, str) or not engine_ref:
            raise TypeError(
                "simulate.rules_by_ref engine_ref must be a non-empty string."
            )
        delegate = self._delegates.get(engine_ref)
        if delegate is None:
            bound = ", ".join(_describe(ref) for ref in self._delegates)
            raise ValueError(
                "simulate.rules_by_ref has no engine bound to engine_ref "
                f"{_describe(engine_ref)}; bound: {bound}."
            )
        return delegate.run(context)


def _describe(engine_ref: str) -> str:
    """Name a reference briefly: its SHA-256 prefix, and its module if any.

    References are often long canonical JSON whose leading keys coincide
    (every Axiom reference starts with the same arithmetic and engine pins), so
    a truncated prefix would not tell two apart.
    """

    digest = hashlib.sha256(engine_ref.encode("utf-8")).hexdigest()[:16]
    try:
        document = json.loads(engine_ref)
    except ValueError:
        document = None
    module = document.get("module") if isinstance(document, dict) else None
    if isinstance(module, str):
        return f"sha256:{digest} (module {module!r})"
    preview = engine_ref if len(engine_ref) <= 60 else engine_ref[:60] + "..."
    return f"sha256:{digest} ({preview!r})"
