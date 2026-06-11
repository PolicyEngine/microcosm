"""A versioned, populace-owned registry of calibration target facts.

The registry is the source of truth for *what the world looks like*: each
:class:`TargetSpec` is one administrative fact (a value, optionally a standard
error, and always a citation) declared independently of any build or scoring
harness. Builds compile a registry into a
:class:`~populace.calibrate.target.TargetSet` (:meth:`TargetRegistry.to_target_set`);
scorers read the same registry — the registry depends on neither, which is the
dependency direction the architecture review demanded (no more harness-extracted
``.npz`` files as the de-facto target store).

Two properties are structural, not stylistic:

- **Versioning is content-addressed.** :attr:`TargetRegistry.version` is a hash
  of the canonical JSON of the specs, so two registries with the same facts have
  the same version and any edit changes it. The JSON artifact embeds the version
  and :meth:`TargetRegistry.from_json` recomputes and compares — a hand-edited
  artifact fails loudly instead of silently steering a calibration.
- **Signs are declared.** A spec must opt in to a negative value with
  ``signed=True``. The net-STCG incident — a positive-only target surface that
  let calibration push short-term capital gains to −$3.9T unnoticed — is the
  motivating case: sign expectations belong in the fact's declaration.

Specs are fully declarative (column names, never callables) so the registry
serializes; a build that needs a computed measure defines the column on its
frame first.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from populace.calibrate.target import Target, TargetSet

__all__ = ["TargetSpec", "TargetRegistry", "specs_from_pe_surface"]

#: Format marker + revision for the JSON artifact.
_FORMAT_KEY = "populace_target_registry"
_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TargetSpec:
    """One declared calibration fact: value, uncertainty, and citation.

    Attributes:
        name: Unique label within ``(name, period)`` (the compiled row label).
        entity: Entity whose weights the fact constrains.
        value: The administrative value to reproduce.
        aggregation: ``"sum"``, ``"count"``, or ``"mean"`` (the
            :class:`~populace.calibrate.target.Target` aggregations).
        measure: Column name read on ``entity``'s table (optional for
            ``count``). Declarative: callables are refused so the registry
            serializes.
        filter: Optional boolean column name gating the records.
        period: Period tag; ``(name, period)`` pairs are distinct facts.
        se: Optional standard error of ``value`` (must be positive when
            given). Recorded for downstream tolerance/weighting decisions;
            not consumed by the loss today.
        source: REQUIRED citation (publication, table, or URL). A fact
            without provenance is not a fact.
        family: Source family for selection/reporting (e.g. ``"irs_soi"``,
            ``"census_population"``).
        signed: Declare that a negative ``value`` is meaningful (e.g. net
            short-term capital gains). Unsigned specs refuse negative values
            — the sign guard.
        tolerance: Optional absolute hit tolerance, passed to the Target.
        notes: Free-text caveats.

    Raises:
        TypeError: If ``measure`` or ``filter`` is not a string (callables
            are not serializable and are refused here by design).
        ValueError: On empty ``name``/``entity``/``source``, a non-finite
            value, a non-positive ``se``, or a negative value without
            ``signed=True``.
    """

    name: str
    entity: str
    value: float
    aggregation: str = "sum"
    measure: str | None = None
    filter: str | None = None
    period: int | str = 0
    se: float | None = None
    source: str = ""
    family: str = "unspecified"
    signed: bool = False
    tolerance: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TargetSpec.name must be non-empty.")
        if not self.entity:
            raise ValueError(f"TargetSpec {self.name!r}: entity must be non-empty.")
        if not self.source:
            raise ValueError(
                f"TargetSpec {self.name!r}: source is required — a fact "
                "without provenance is not a fact."
            )
        for label, column in (("measure", self.measure), ("filter", self.filter)):
            if column is not None and not isinstance(column, str):
                raise TypeError(
                    f"TargetSpec {self.name!r}: {label} must be a column name "
                    f"(str) or None — callables do not serialize; got "
                    f"{type(column).__name__}."
                )
        if not math.isfinite(self.value):
            raise ValueError(
                f"TargetSpec {self.name!r}: value must be finite, got {self.value!r}."
            )
        if self.se is not None and not (self.se > 0):
            raise ValueError(
                f"TargetSpec {self.name!r}: se must be positive when given, "
                f"got {self.se!r}."
            )
        if self.value < 0 and not self.signed:
            raise ValueError(
                f"TargetSpec {self.name!r}: negative value {self.value!r} "
                "requires signed=True — declare the sign expectation "
                "explicitly (the net-STCG lesson)."
            )

    @property
    def key(self) -> tuple[str, int | str]:
        """The ``(name, period)`` identity of the fact."""
        return (self.name, self.period)

    def to_target(self) -> Target:
        """Compile the spec into a calibration :class:`Target`."""
        return Target(
            name=self.name,
            entity=self.entity,
            measure=self.measure,
            aggregation=self.aggregation,
            value=self.value,
            period=self.period,
            tolerance=self.tolerance,
            filter=self.filter,
            source=self.source,
        )


class TargetRegistry:
    """An ordered, content-addressed collection of :class:`TargetSpec` facts.

    Args:
        specs: The facts, in order. Duplicate ``(name, period)`` keys are
            rejected.
        country: The jurisdiction the registry describes (e.g. ``"us"``).

    Raises:
        TypeError: If an entry is not a :class:`TargetSpec`.
        ValueError: On duplicate ``(name, period)`` keys or an empty country.
    """

    def __init__(self, specs: Iterable[TargetSpec], *, country: str) -> None:
        if not country:
            raise ValueError("TargetRegistry.country must be non-empty.")
        materialized = tuple(specs)
        seen: set[tuple[str, int | str]] = set()
        for spec in materialized:
            if not isinstance(spec, TargetSpec):
                raise TypeError(
                    "TargetRegistry entries must be TargetSpec instances, "
                    f"got {type(spec).__name__}."
                )
            if spec.key in seen:
                raise ValueError(
                    f"Duplicate target key {spec.key!r}: a (name, period) "
                    "pair may be declared only once."
                )
            seen.add(spec.key)
        self._specs = materialized
        self._country = country

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def specs(self) -> tuple[TargetSpec, ...]:
        return self._specs

    @property
    def country(self) -> str:
        return self._country

    @property
    def version(self) -> str:
        """Content hash of the canonical spec JSON (first 12 hex chars).

        Same facts -> same version; any value/source/sign edit -> a new
        version. This is the identifier release manifests pin.
        """
        canonical = json.dumps(
            {
                "country": self._country,
                "specs": [asdict(spec) for spec in self._specs],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[TargetSpec]:
        return iter(self._specs)

    def __repr__(self) -> str:
        return (
            f"TargetRegistry(country={self._country!r}, n={len(self._specs)}, "
            f"version={self.version!r})"
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def families(self) -> tuple[str, ...]:
        """Sorted distinct source families."""
        return tuple(sorted({spec.family for spec in self._specs}))

    def select(
        self,
        *,
        family: str | None = None,
        predicate: Callable[[TargetSpec], bool] | None = None,
    ) -> TargetRegistry:
        """A sub-registry of the specs matching ``family`` and ``predicate``."""
        chosen = [
            spec
            for spec in self._specs
            if (family is None or spec.family == family)
            and (predicate is None or predicate(spec))
        ]
        return TargetRegistry(chosen, country=self._country)

    # ------------------------------------------------------------------
    # Compilation + serialization
    # ------------------------------------------------------------------

    def to_target_set(self) -> TargetSet:
        """Compile every spec into a :class:`TargetSet` for the solver."""
        return TargetSet(tuple(spec.to_target() for spec in self._specs))

    def to_json(self, path: str | Path) -> Path:
        """Write the versioned JSON artifact and return its path."""
        path = Path(path)
        payload = {
            _FORMAT_KEY: _FORMAT_VERSION,
            "country": self._country,
            "version": self.version,
            "n_specs": len(self._specs),
            "specs": [asdict(spec) for spec in self._specs],
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> TargetRegistry:
        """Load a registry artifact, verifying its embedded content version.

        Raises:
            ValueError: If the file is not a registry artifact, its format
                revision is unknown, or its embedded version does not match
                the recomputed content hash (a tampered/hand-edited
                artifact).
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get(_FORMAT_KEY) != _FORMAT_VERSION:
            raise ValueError(
                f"{path} is not a populace target registry artifact "
                f"(format revision {_FORMAT_VERSION})."
            )
        specs = tuple(TargetSpec(**raw) for raw in payload["specs"])
        registry = cls(specs, country=payload["country"])
        stored = payload.get("version")
        if stored != registry.version:
            raise ValueError(
                f"Registry artifact {path} is corrupt: embedded version "
                f"{stored!r} does not match the recomputed content hash "
                f"{registry.version!r}. Regenerate the artifact instead of "
                "editing it by hand."
            )
        return registry


def specs_from_pe_surface(
    names: Iterable[str],
    values: Iterable[float],
    *,
    family_sources: Mapping[str, str],
    entity: str = "household",
    period: int | str = 0,
    signed_names: Iterable[str] = (),
    measures: Mapping[str, str] | None = None,
) -> tuple[TargetSpec, ...]:
    """Ingest an extracted PE-native target surface into registry specs.

    The transitional path from harness-extracted surfaces to declared facts:
    each ``name`` is parsed as ``"family/rest"`` (the PE-native convention);
    the family's citation is looked up in ``family_sources``. Unknown
    families are refused — every ingested fact must land with provenance.

    Args:
        names: Target row names (``"family/..."``).
        values: Target values, aligned with ``names``.
        family_sources: REQUIRED citation per family prefix. A name whose
            family is missing here raises.
        entity: Entity the rows constrain (extracted surfaces are
            household-grain today).
        period: Period tag for every spec.
        signed_names: Names whose negative values are meaningful.
        measures: Optional ``name -> measure column`` mapping. Extracted
            surfaces precompile their rows, so most specs carry no measure;
            a build recompiling from the registry supplies them here.

    Returns:
        One :class:`TargetSpec` per name, in input order.

    Raises:
        ValueError: If lengths differ, a name has no family prefix, or a
            family has no source.
    """
    names = list(names)
    values = list(values)
    if len(names) != len(values):
        raise ValueError(
            f"names and values must align: {len(names)} names vs {len(values)} values."
        )
    signed = set(signed_names)
    measures = dict(measures or {})
    specs: list[TargetSpec] = []
    for name, value in zip(names, values, strict=True):
        family = name.split("/", 1)[0] if "/" in name else ""
        if not family:
            raise ValueError(
                f"Target name {name!r} has no 'family/...' prefix; extracted "
                "surfaces must carry their source family in the name."
            )
        if family not in family_sources:
            raise ValueError(
                f"Family {family!r} (target {name!r}) has no source in "
                "family_sources — every ingested fact needs provenance."
            )
        specs.append(
            TargetSpec(
                name=name,
                entity=entity,
                value=float(value),
                measure=measures.get(name),
                period=period,
                source=family_sources[family],
                family=family,
                signed=name in signed,
            )
        )
    return tuple(specs)
