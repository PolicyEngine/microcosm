"""The RulesEngine protocol: rules engines are adapters, not dependencies.

A rules engine (policyengine-us today, Axiom rulespec-us when it lands)
resolves variable metadata, materializes computed variables onto a bundle,
and writes bundles out as engine-native datasets. Nothing outside an adapter
imports a rules engine; the kernel and every operator talk to this protocol.

:class:`ExportContract` is the frozen column-parity contract a dataset export
is gated against: which columns it must contain, must not contain, may carry,
and must leave to the engine's own formulas.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from populace.frame.bundle import Frame
from populace.frame.schema import EntitySchema

__all__ = ["RulesEngine", "ExportContract"]


@runtime_checkable
class RulesEngine(Protocol):
    """Adapter interface to a tax-benefit rules engine."""

    def variable_entity(self, name: str) -> str:
        """Return the entity a variable lives on (e.g. ``"tax_unit"``)."""
        ...

    def variable_dtype(self, name: str) -> type:
        """Return the value type of a variable (e.g. ``float``)."""
        ...

    def entity_schema(self) -> EntitySchema:
        """Return the engine's entity structure as an :class:`EntitySchema`."""
        ...

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        """Compute ``variables`` for ``period`` on the bundle.

        Returns:
            One array per variable, row-aligned to the variable's entity
            table.
        """
        ...

    def export_contract(self) -> "ExportContract":
        """Return the column-parity contract exports are gated against."""
        ...

    def write_dataset(
        self,
        bundle: Frame,
        path: str | Path,
        period: int | str,
    ) -> None:
        """Write the bundle as an engine-native dataset at ``path``."""
        ...


@dataclass(frozen=True)
class ExportContract:
    """Frozen column-parity contract for a rules-engine dataset export.

    Attributes:
        required: Columns the export MUST contain. A missing required column
            fails the export gate.
        forbidden: Columns the export MUST NOT contain. They are dropped on
            sight and their presence fails the gate.
        optional: Bookkeeping columns that are neither required nor
            forbidden; an export passes them through untouched if present.
        formula_owned_excluded: Variables the engine owns through formulas
            and the baseline does not persist as inputs. They are silently
            dropped if present so the engine computes them itself.
    """

    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    optional: tuple[str, ...]
    formula_owned_excluded: tuple[str, ...]

    @classmethod
    def empty(cls) -> "ExportContract":
        """Return a contract with no constraints (everything passes)."""
        return cls(required=(), forbidden=(), optional=(), formula_owned_excluded=())

    @classmethod
    def from_path(cls, path: str | Path) -> "ExportContract":
        """Load a contract from a JSON manifest.

        Keys whose name starts with ``"_"`` (documentation/metadata such as
        ``_description`` and ``_categories``) are ignored. The optional
        section is read from ``"optional"``, falling back to the
        ``"ecps_internal_optional"`` key used by existing eCPS parity
        manifests.

        Args:
            path: Filesystem path to the contract JSON manifest.

        Returns:
            The parsed :class:`ExportContract`.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = {
            key: value for key, value in payload.items() if not key.startswith("_")
        }
        optional = sections.get("optional", sections.get("ecps_internal_optional", ()))
        return cls(
            required=_as_str_tuple(sections.get("required", ())),
            forbidden=_as_str_tuple(sections.get("forbidden", ())),
            optional=_as_str_tuple(optional),
            formula_owned_excluded=_as_str_tuple(
                sections.get("formula_owned_excluded", ())
            ),
        )


def _as_str_tuple(values: Any) -> tuple[str, ...]:
    """Coerce a JSON list (or any iterable) into a tuple of strings."""
    if values is None:
        return ()
    return tuple(str(value) for value in values)
