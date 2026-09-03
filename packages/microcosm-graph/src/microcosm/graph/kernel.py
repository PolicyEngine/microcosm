"""The kernel protocol: pure functions over typed slices.

A kernel never sees the population. It receives read-only views of the
slices its node declared, the node's parameters, an RNG seeded from the
node key, and content-verified paths for the sources it declared. It
returns values for the cells its node owns (indexed by entity id), and the
executor enforces everything else: ownership, dtype, immutability, mass,
and receipts.

Structural kernels return data, and the executor does the structural work:

- ``CREATE`` is the one kernel that builds a population; it returns
  :attr:`KernelResult.frame`.
- ``FILTER`` returns the surviving-row mask as :attr:`KernelResult.keep`;
  the executor subsets the base version by id, carries every column, and
  records mass.
- ``EXPAND`` returns the clone lineage as :attr:`KernelResult.expand` (per
  entity, new ids to the source ids they copy) plus the new weights; the
  executor carries every column from the source rows, records the lineage
  in the receipt, and records mass. A node declared ``entrants=True`` may
  also add rows with null lineage; the kernel then materializes their
  columns, and for entrant persons their stratum through
  :attr:`KernelResult.strata` (amendments 11 and 14).
- ``REWEIGHT`` (and any node with a declared weight transition) returns
  :attr:`KernelResult.weights`; the executor validates the kind transition
  and the mass policy.

No other kernel ever holds a population (charter B2).

A kernel's :class:`Capabilities` also declare its :class:`KernelRole`: an
ordinary computation, a gate (its receipt carries one of
:data:`~microcosm.graph.decl.GATE_OUTCOMES`), or a release (its owned tier
is derived from the gate verdicts in its ancestry, and its receipt reports
``unreached`` when a required human decision is absent from the run).

Numbers carry their own contract. A kernel whose :class:`Numeric` claim is
``tolerance_bound`` declares a :class:`Tolerance`; the executor records it
in the receipt and hands every reader the declared tolerance of each input
cell's owner through :attr:`KernelContext.tolerances`, so a gate compares
against a declaration rather than a guess (amendment 13).

This file is a frozen interface (see ``docs/graph-acceptance.md``).
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microcosm.frame import Frame, Weights

from .decl import Node, Param, StructuralDelta

__all__ = [
    "Capabilities",
    "Determinism",
    "Kernel",
    "KernelBase",
    "KernelContext",
    "KernelRegistry",
    "KernelResult",
    "KernelRole",
    "Numeric",
    "SeedSource",
    "Tolerance",
    "source_hash",
]


class Determinism(StrEnum):
    """Whether a kernel's output is a function of its inputs and seed."""

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    NONDETERMINISTIC = "nondeterministic"


class Numeric(StrEnum):
    """How reproducible a kernel's numbers are across runs.

    ``bitwise``: identical bytes on every platform. ``platform_bitwise``:
    identical bytes on one platform (architecture and locked dependencies),
    with no bound on how far a cell may move across platforms; a quantile
    forest is the model case, where a one-ulp difference can flip which
    donor a draw lands on (amendment 16). ``tolerance_bound``: every cell
    within a declared :class:`Tolerance` across platforms.
    """

    BITWISE = "bitwise"
    PLATFORM_BITWISE = "platform_bitwise"
    TOLERANCE_BOUND = "tolerance_bound"


@dataclass(frozen=True)
class Tolerance:
    """How far a ``tolerance_bound`` kernel's numbers may move between runs.

    Two values agree when they are within ``atol`` absolutely, or within
    ``rtol`` relatively, or within ``ulps`` last-place units of each other.
    A bitwise kernel declares no tolerance at all.

    Attributes:
        rtol: Relative tolerance; non-negative and finite.
        atol: Absolute tolerance; non-negative and finite.
        ulps: Units in the last place; non-negative.
    """

    rtol: float = 0.0
    atol: float = 0.0
    ulps: int = 0

    def __post_init__(self) -> None:
        for name in ("rtol", "atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"Tolerance.{name} must be a number.")
            try:
                # Keys and manifests carry the float; an integer too large for
                # one would only fail later, at identity time.
                as_float = float(value)
            except OverflowError as error:
                raise ValueError(
                    f"Tolerance.{name} must be representable as a finite float."
                ) from error
            if not (as_float >= 0.0) or as_float == float("inf"):
                raise ValueError(f"Tolerance.{name} must be non-negative and finite.")
            object.__setattr__(self, name, as_float)
        if isinstance(self.ulps, bool) or not isinstance(self.ulps, int):
            raise ValueError("Tolerance.ulps must be an integer.")
        if self.ulps < 0:
            raise ValueError("Tolerance.ulps must be non-negative.")
        if self.rtol == 0.0 and self.atol == 0.0 and self.ulps == 0:
            raise ValueError(
                "Tolerance must allow some movement; a bitwise kernel declares "
                "no tolerance instead."
            )


class SeedSource(StrEnum):
    """Where a seeded kernel takes its randomness from."""

    EXECUTOR = "executor"  # ``KernelContext.rng``, derived from the node key
    PARAM = "param"  # a literal ``seed`` parameter (legacy parity kernels)
    NONE = "none"


class KernelRole(StrEnum):
    """What a kernel's node means to the release process."""

    COMPUTE = "compute"
    GATE = "gate"  # receipt["outcome"] is one of GATE_OUTCOMES
    RELEASE = "release"  # owns a tier derived from gate ancestry


@dataclass(frozen=True)
class Capabilities:
    """A kernel's declared contract, recorded in every receipt.

    Attributes:
        determinism: See :class:`Determinism`.
        numeric: See :class:`Numeric`.
        seed_source: See :class:`SeedSource`.
        structural: The row-set change the kernel performs; must match the
            node's declaration.
        role: See :class:`KernelRole`.
        consumes_se: Whether a calibration kernel uses declared target
            standard errors. A kernel that ignores them says so here.
        dependencies: Installed distributions whose versions enter the
            implementation hash.
        tolerance: Required when ``numeric`` is ``tolerance_bound`` and
            forbidden otherwise: how far the kernel's numbers may move
            between runs or machines.
    """

    determinism: Determinism
    numeric: Numeric = Numeric.BITWISE
    seed_source: SeedSource = SeedSource.NONE
    structural: StructuralDelta = StructuralDelta.NONE
    role: KernelRole = KernelRole.COMPUTE
    consumes_se: bool = False
    dependencies: tuple[str, ...] = ()
    tolerance: Tolerance | None = None

    def __post_init__(self) -> None:
        # Every field is validated here, so a registered contract is a real
        # one: a string that spells an enum member does not pass as the member.
        for name, kind in (
            ("determinism", Determinism),
            ("numeric", Numeric),
            ("seed_source", SeedSource),
            ("structural", StructuralDelta),
            ("role", KernelRole),
        ):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f"Capabilities.{name} must be a {kind.__name__}.")
        if not isinstance(self.consumes_se, bool):
            raise TypeError("Capabilities.consumes_se must be a boolean.")
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(name, str) or not name for name in self.dependencies
        ):
            raise TypeError(
                "Capabilities.dependencies must be a tuple of distribution names."
            )
        if self.tolerance is not None and not isinstance(self.tolerance, Tolerance):
            raise TypeError("Capabilities.tolerance must be a Tolerance or None.")
        if self.numeric is Numeric.TOLERANCE_BOUND and self.tolerance is None:
            raise ValueError(
                "A tolerance_bound kernel must declare its Tolerance; a claim of "
                "bounded movement without a bound is not a claim."
            )
        if (
            self.numeric in (Numeric.BITWISE, Numeric.PLATFORM_BITWISE)
            and self.tolerance is not None
        ):
            raise ValueError("A bitwise kernel declares no Tolerance.")


@dataclass(frozen=True)
class NumericScope:
    """The numeric contract a gate may hold one input coordinate to.

    The executor derives it from the coordinate's writers under the
    loosest-writer rule, ordered ``bitwise`` < ``platform_bitwise`` <
    ``tolerance_bound``. A platform-bitwise writer never disappears into
    a bound: it leaves ``platform`` set, which says the contract holds on
    that platform only and that a cross-platform comparison has no bound
    (amendment 17).

    Attributes:
        numeric: The loosest :class:`Numeric` class among the writers.
        tolerance: The loosest declared :class:`Tolerance` among the
            ``tolerance_bound`` writers; ``None`` for the other classes.
        platform: The platform fingerprint the contract holds on, or
            ``None`` when it holds on every platform. Required when
            ``numeric`` is ``platform_bitwise``; permitted on
            ``tolerance_bound`` when a platform-bitwise writer contributed;
            forbidden on ``bitwise``.
    """

    numeric: Numeric = Numeric.BITWISE
    tolerance: Tolerance | None = None
    platform: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.numeric, Numeric):
            raise ValueError("NumericScope.numeric must be a Numeric.")
        if self.tolerance is not None and not isinstance(self.tolerance, Tolerance):
            raise ValueError("NumericScope.tolerance must be a Tolerance or None.")
        if self.platform is not None and (
            not isinstance(self.platform, str) or not self.platform
        ):
            raise ValueError("NumericScope.platform must be a non-empty str or None.")
        if self.numeric is Numeric.TOLERANCE_BOUND and self.tolerance is None:
            raise ValueError("A tolerance_bound scope must carry its Tolerance.")
        if self.numeric is not Numeric.TOLERANCE_BOUND and self.tolerance is not None:
            raise ValueError(f"A {self.numeric.value} scope carries no Tolerance.")
        if self.numeric is Numeric.PLATFORM_BITWISE and self.platform is None:
            raise ValueError("A platform_bitwise scope must name its platform.")
        if self.numeric is Numeric.BITWISE and self.platform is not None:
            raise ValueError("A bitwise scope holds on every platform.")


@dataclass(frozen=True)
class KernelContext:
    """Everything a kernel may read. Nothing here is writable.

    Attributes:
        node: The declaration being executed.
        tables: Entity name to a read-only view holding the entity id
            column(s) plus exactly the declared input columns, restricted to
            the declared row mask. Every entity the node owns cells on is
            present at least as an id-only view (with membership columns on
            the person entity), so a kernel can index what it is
            responsible for.
        weights: Entity name to effective typed weights, for entities named
            in the node's inputs or outputs.
        strata: Read-only per-person strata of the population version.
        params: The node's parameters.
        rng: A generator seeded from the node key. The only randomness a
            kernel may use.
        sources: Source name to a content-verified path, for declared
            sources only.
        tolerances: ``(entity, column)`` of each declared input column to
            the :class:`Tolerance` its owning kernel declared, or ``None``
            for a bitwise owner. A gate compares against these.
        numerics: ``(entity, column)`` of each declared input column to
            its :class:`NumericScope`; ``tolerances`` is its projection
            (``numerics[c].tolerance == tolerances[c]``). A gate that
            compares across platforms consults the scope's ``platform``
            (amendment 17).
    """

    node: Node
    tables: Mapping[str, pd.DataFrame]
    weights: Mapping[str, Weights]
    strata: pd.Series
    params: Mapping[str, Param]
    rng: np.random.Generator
    sources: Mapping[str, Path] = field(default_factory=dict)
    tolerances: Mapping[tuple[str, str], Tolerance | None] = field(default_factory=dict)
    numerics: Mapping[tuple[str, str], NumericScope] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelResult:
    """What a kernel returns. The executor validates every field.

    Attributes:
        columns: ``(entity, column)`` to a Series indexed by the entity ids
            of the owned positions. Extra ids, missing ids, or a dtype other
            than the declared one reject the node.
        frame: ``CREATE`` kernels only: the new population version.
        keep: ``FILTER`` kernels only: a boolean Series indexed by the ids
            of the filtered entity in the base version; ``True`` keeps the
            row. The executor applies it and records mass.
        expand: ``EXPAND`` kernels only: entity name to a Series indexed by
            the ids of the rows the new version adds, whose values are the
            ids of the base rows they copy. Every base row survives; the
            executor carries each column from the source row, and the
            person-to-group memberships of a copied group's members follow
            the copied group. Weights for expanded entities come through
            ``weights``.
        weights: ``REWEIGHT`` kernels and declared weight transitions only:
            the new explicit weights of the transition's entity.
        strata: ``EXPAND`` kernels on a node with ``entrants=True`` only: the
            stratum label of every entrant person, indexed by its new id.
            Copied persons inherit their source's stratum and must not
            appear here; an entrant person absent from it rejects the node.
        artifacts: Opaque bytes stored beside the node's outputs (a fitted
            model, a diagnostic table), keyed by name.
        receipt: Descriptive facts for the manifest. Never hashed into a
            key. A gate kernel puts its verdict under ``"outcome"`` and its
            evidence under ``"evidence"``; a mass-changing kernel may put
            its own accounting under ``"mass"`` (the executor records its
            own regardless).
    """

    columns: Mapping[tuple[str, str], pd.Series] = field(default_factory=dict)
    frame: Frame | None = None
    keep: pd.Series | None = None
    expand: Mapping[str, pd.Series] | None = None
    weights: Weights | None = None
    artifacts: Mapping[str, bytes] = field(default_factory=dict)
    receipt: Mapping[str, object] = field(default_factory=dict)
    strata: pd.Series | None = None


@runtime_checkable
class Kernel(Protocol):
    """A registered computation.

    Attributes:
        ref: The reference nodes name, e.g. ``"fit.qrf@1"``. The suffix is
            the kernel's contract version; a behavior change bumps it.
        capabilities: The kernel's declared contract.
    """

    ref: str
    capabilities: Capabilities

    def implementation_hash(self) -> str:
        """SHA-256 over the kernel's source and its declared dependency versions."""
        ...

    def run(self, context: KernelContext) -> KernelResult:
        """Compute the node. Must not mutate anything in ``context``."""
        ...


def source_hash(
    *objects: ModuleType | type | Callable[..., object],
    dependencies: tuple[str, ...] = (),
) -> str:
    """Hash the source files behind ``objects`` plus dependency versions.

    The digest is over the bytes of each object's defining module, so any
    edit to that module changes it, and over the installed version string of
    each dependency, so a behavior-bearing upgrade changes it too.
    """

    digest = hashlib.sha256(b"microcosm-graph/source-hash/1\n")
    seen: set[str] = set()
    for obj in objects:
        module = obj if isinstance(obj, ModuleType) else inspect.getmodule(obj)
        if module is None or module.__file__ is None:
            raise ValueError(f"Cannot locate source for {obj!r}.")
        path = Path(module.__file__).resolve()
        if str(path) in seen:
            continue
        seen.add(str(path))
        if not path.is_file():
            raise ValueError(
                f"{obj!r} has no source file on disk ({path}); a kernel defined "
                "interactively cannot carry an implementation hash."
            )
        content = path.read_bytes()
        digest.update(module.__name__.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    for distribution in sorted(dependencies):
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as error:
            raise ValueError(
                f"Dependency {distribution!r} is declared by a kernel but is not "
                "installed; an implementation hash cannot omit it."
            ) from error
        digest.update(f"{distribution}=={version}\n".encode())
    return digest.hexdigest()


class KernelBase:
    """Default ``implementation_hash``: this class's module plus dependencies.

    Subclasses set ``ref`` and ``capabilities`` and implement ``run``.
    """

    ref: str
    capabilities: Capabilities

    def implementation_hash(self) -> str:
        return source_hash(type(self), dependencies=self.capabilities.dependencies)


class KernelRegistry:
    """Kernels by reference. The executor resolves nodes through one registry."""

    def __init__(self) -> None:
        self._kernels: dict[str, Kernel] = {}

    def register(self, kernel: Kernel) -> Kernel:
        if not isinstance(kernel, Kernel):
            raise TypeError(f"{kernel!r} does not satisfy the Kernel protocol.")
        if not isinstance(kernel.capabilities, Capabilities):
            raise TypeError(
                f"Kernel {getattr(kernel, 'ref', kernel)!r} must carry a Capabilities "
                "instance, not a look-alike."
            )
        if kernel.ref in self._kernels and self._kernels[kernel.ref] is not kernel:
            raise ValueError(f"Kernel {kernel.ref!r} is already registered.")
        self._kernels[kernel.ref] = kernel
        return kernel

    def get(self, ref: str) -> Kernel:
        try:
            return self._kernels[ref]
        except KeyError as error:
            raise KeyError(f"No kernel registered as {ref!r}.") from error

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._kernels))

    def implementation_hash(self, ref: str) -> str:
        return self.get(ref).implementation_hash()

    def as_mapping(self) -> Mapping[str, Kernel]:
        return MappingProxyType(dict(self._kernels))
