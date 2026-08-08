"""The RulesEngine protocol: rules engines are adapters, not dependencies.

A rules engine (policyengine-us today, Axiom rulespec-us when it lands)
resolves variable metadata, materializes computed variables onto a bundle,
and writes bundles out as engine-native datasets. Nothing outside an adapter
imports a rules engine; the kernel and every operator talk to this protocol.

:class:`ExportContract` is the frozen column-parity contract a dataset export
is gated against: which columns it must contain, must not contain, may carry,
must leave to the engine's own formulas, and whether the surface is closed to
unexpected extras.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from microcosm.frame.bundle import Frame
from microcosm.frame.schema import EntitySchema, VariableMetadata

__all__ = ["RulesEngine", "ExportContract"]


@runtime_checkable
class RulesEngine(Protocol):
    """Adapter interface to a tax-benefit rules engine.

    The protocol is deliberately minimal: it resolves variable metadata,
    enumerates input variables, materializes computed variables onto a
    frame, and writes a frame as an engine-native dataset. An adapter wraps
    one engine (policyengine-us today, Axiom rulespec-us next); nothing
    outside an adapter imports a rules engine.

    Known future extensions, called out so the additions are planned rather
    than discovered (each is a breaking protocol change when it lands):

    - **Reform / branch simulation.** Any policy-comparison evaluation needs
      to materialize variables under a counterfactual parameter set. Deferred;
      will arrive as a ``materialize(..., reform=...)`` parameter or a sibling
      method.
    - **Multi-period materialization.** ``materialize`` is single-period;
      longitudinal runs will need a period range. Deferred.
    - **Enum value domains and defaults.** ``variable_metadata`` reports a
      coarse dtype kind; full enum value sets and default values are not yet
      exposed.
    """

    def variable_metadata(self, name: str) -> VariableMetadata:
        """Return the variable's owning entity, dtype kind, and period.

        Replaces the older ``variable_entity`` / ``variable_dtype`` pair so a
        single call resolves everything the kernel needs about a variable,
        including the period semantics the charter promises.
        """
        ...

    def variables(self) -> Sequence[str]:
        """Return the input variables the engine accepts on a dataset.

        The spec engine needs to enumerate inputs (to know what a pool must
        produce); this is that surface. Computed/formula-owned outputs are
        not included.
        """
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
        forbidden: Columns the export MUST NOT contain. Their presence fails
            the export gate.
        optional: Bookkeeping columns that are neither required nor
            forbidden; an export passes them through untouched if present.
        formula_owned_excluded: Variables the engine owns through formulas
            and the baseline must not persist as inputs. Their presence fails
            the export gate; upstream build stages must drop them before
            calling the engine writer.
        closed: If true, exports may only contain required/optional columns
            plus adapter-defined structural columns. Unexpected extras fail
            the export gate before anything is written.
    """

    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    optional: tuple[str, ...]
    formula_owned_excluded: tuple[str, ...]
    closed: bool = False

    @classmethod
    def empty(cls) -> "ExportContract":
        """Return a contract with no constraints (everything passes)."""
        return cls(required=(), forbidden=(), optional=(), formula_owned_excluded=())

    @classmethod
    def from_path(cls, path: str | Path) -> "ExportContract":
        """Load a contract from a JSON manifest.

        Keys whose name starts with ``"_"`` (documentation/metadata such as
        ``_description`` and ``_categories``) are ignored. The optional
        section is read from ``"optional"``.

        Args:
            path: Filesystem path to the contract JSON manifest.

        Returns:
            The parsed :class:`ExportContract`.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sections = {
            key: value for key, value in payload.items() if not key.startswith("_")
        }
        optional = sections.get("optional", ())
        return cls(
            required=_as_str_tuple(sections.get("required", ())),
            forbidden=_as_str_tuple(sections.get("forbidden", ())),
            optional=_as_str_tuple(optional),
            formula_owned_excluded=_as_str_tuple(
                sections.get("formula_owned_excluded", ())
            ),
            closed=bool(sections.get("closed", False)),
        )


def _as_str_tuple(values: Any) -> tuple[str, ...]:
    """Coerce a JSON list (or any iterable) into a tuple of strings."""
    if values is None:
        return ()
    return tuple(str(value) for value in values)
