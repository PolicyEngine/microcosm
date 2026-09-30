"""Committed, engine-generated snapshots of Axiom RuleSpec input surfaces.

An Axiom input is any name a compiled RuleSpec rule references but does not
derive, scoped to the module compiled and to the engine entity whose program
reads it (see :mod:`microcosm.frame.adapters.axiom`). CI has no Axiom
engine, so ``tools/refresh_axiom_input_surface.py`` compiles every module of
a rulespec country at a pinned commit with the real engine and commits the
result as a test fixture, ``packages/microcosm-frame/tests/fixtures/
axiom_input_surfaces/<country>.json``. The concept-mapping tests and the
coverage reports read these snapshots through this parser; a snapshot is
evidence of what one engine build compiled from one RuleSpec commit, nothing
more.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

from microcosm.frame.concept_mapping import InputRef

__all__ = [
    "AXIOM_INPUT_SURFACE_FORMAT",
    "AxiomInputSurface",
    "AxiomModuleSurface",
    "load_axiom_input_surface",
]

AXIOM_INPUT_SURFACE_FORMAT = "microcosm.axiom_input_surface.v1"


@dataclass(frozen=True)
class AxiomModuleSurface:
    """One RuleSpec module's compiled input surface.

    Attributes:
        path: The module's path relative to its rulespec root.
        sha256: SHA-256 of the module file's bytes at the pinned commit.
        status: ``compiled``, or why the module has no surface.
        inputs: Engine entity -> the root input names its program reads.
        canonical_inputs: Input name -> the engine's canonical request name.
    """

    path: str
    sha256: str
    status: str
    inputs: Mapping[str, tuple[str, ...]]
    canonical_inputs: Mapping[str, str | None]

    def refs(self) -> tuple[InputRef, ...]:
        """Every (name, entity, module) input this module accepts."""
        return tuple(
            InputRef(name, entity, self.path)
            for entity, names in sorted(self.inputs.items())
            for name in names
        )


@dataclass(frozen=True)
class AxiomInputSurface:
    """A rulespec country's input surface at one pinned commit.

    Attributes:
        country: The rulespec country code.
        rulespec_commit: The rulespec commit compiled.
        engine_commit: The axiom-rules-engine commit that compiled it.
        engine_surface: Which engine surface produced the inputs.
        modules: Module path -> its surface.
    """

    country: str
    rulespec_commit: str
    engine_commit: str
    engine_surface: str
    modules: Mapping[str, AxiomModuleSurface]

    def refs(self) -> tuple[InputRef, ...]:
        """Every module-scoped input in the country, sorted."""
        return tuple(
            sorted(ref for module in self.modules.values() for ref in module.refs())
        )

    def canonical_input(self, ref: InputRef) -> str | None:
        """The engine's canonical request name for ``ref``."""
        module = self.modules[ref.module]
        return module.canonical_inputs.get(ref.name)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AxiomInputSurface:
        """Parse a snapshot, refusing an unknown format."""
        if data.get("format") != AXIOM_INPUT_SURFACE_FORMAT:
            raise ValueError(
                f"An Axiom input surface must declare format "
                f"{AXIOM_INPUT_SURFACE_FORMAT!r}."
            )
        modules = {}
        for entry in data["modules"]:
            inputs = {
                entity: tuple(names)
                for entity, names in entry.get("inputs", {}).items()
            }
            modules[entry["path"]] = AxiomModuleSurface(
                path=entry["path"],
                sha256=entry["sha256"],
                status=entry["status"],
                inputs=MappingProxyType(inputs),
                canonical_inputs=MappingProxyType(
                    dict(entry.get("canonical_inputs", {}))
                ),
            )
        return cls(
            country=data["country"],
            rulespec_commit=data["rulespec"]["commit"],
            engine_commit=data["engine"]["commit"],
            engine_surface=data["engine"]["surface"],
            modules=MappingProxyType(modules),
        )


@cache
def load_axiom_input_surface(path: str | Path) -> AxiomInputSurface:
    """Load one committed input-surface snapshot from ``path``."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AxiomInputSurface.from_dict(data)
