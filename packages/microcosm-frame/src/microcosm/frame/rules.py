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
import pandas as pd

from microcosm.frame.bundle import Frame
from microcosm.frame.schema import EntitySchema, VariableMetadata

__all__ = [
    "RulesEngine",
    "ExportContract",
    "EngineInput",
    "InputInventory",
    "InputInventoryProvider",
    "assert_rules_engine_country",
    "materialize_rules_engine_predictors",
]


@dataclass(frozen=True)
class EngineInput:
    """Observed executable input, not a complete population concept.

    ``entity`` is the adapter's operational frame entity. A request address
    identifies an executable input; it does not establish semantic equivalence
    to a publisher concept. Missing authored metadata remains ``None``.
    """

    name: str
    entity: str
    engine_entity: str
    canonical_request_name: str | None
    request_names: tuple[str, ...]
    dtype: str | None = None
    unit: str | None = None
    period: str | None = None
    definition: str | None = None
    concept_id: str | None = None
    required: bool | None = None


@dataclass(frozen=True)
class InputInventory:
    """A module-scoped input observation with content fingerprints.

    This is not an export contract or a closed population schema. Inputs
    outside the mapped entity surface and future-reform fields are not
    adjudicated by this diagnostic.
    """

    inputs: tuple[EngineInput, ...]
    fingerprints: tuple[Mapping[str, str], ...]
    mapped_entities: tuple[str, ...]
    entity_discovery: tuple[Mapping[str, Any], ...]
    runtime: Mapping[str, str | None]


class InputInventoryProvider(Protocol):
    """Optional metadata discovery; existing RulesEngine adapters need not opt in."""

    def input_inventory(self) -> InputInventory:
        """Discover executable inputs without guessing missing semantics."""
        ...


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


def assert_rules_engine_country(engine: RulesEngine, country: str) -> None:
    """Require a rules adapter to declare the dataset country it serves."""

    engine_country = getattr(engine, "country", None)
    if not isinstance(engine_country, str) or not engine_country:
        raise ValueError("Rules engine adapter must declare a non-empty country.")
    if engine_country != country:
        raise ValueError(
            f"Rules engine country {engine_country!r} does not match dataset "
            f"country {country!r}."
        )


def materialize_rules_engine_predictors(
    bundle: Frame,
    *,
    variables: Sequence[str],
    period: int | str,
    engine: RulesEngine,
    country: str | None = None,
) -> Frame:
    """Return ``bundle`` with rules-engine predictors materialized as columns."""

    if country is not None:
        assert_rules_engine_country(engine, country)
    requested = tuple(variables)
    if not requested:
        raise ValueError("materialize_rules_engine_predictors requires variables.")
    existing: list[str] = []
    for variable in requested:
        try:
            bundle.column_entity(variable)
        except ValueError:
            continue
        existing.append(variable)
    if existing:
        raise ValueError(
            "Cannot materialize rules-engine predictor(s) already present on "
            f"the frame: {existing}."
        )

    materialized = engine.materialize(bundle, requested, period)
    tables = {entity: bundle.table(entity).copy() for entity in bundle.entities}
    for variable in requested:
        if variable not in materialized:
            raise ValueError(
                f"Rules engine did not return materialized predictor {variable!r}."
            )
        metadata = engine.variable_metadata(variable)
        if metadata.entity not in tables:
            raise ValueError(
                f"Materialized predictor {variable!r} belongs to entity "
                f"{metadata.entity!r}, which is not in the frame."
            )
        values = np.asarray(materialized[variable])
        expected = bundle.n(metadata.entity)
        if values.shape != (expected,):
            raise ValueError(
                f"Materialized predictor {variable!r} has shape {values.shape} "
                f"but entity {metadata.entity!r} has {expected} row(s)."
            )
        tables[metadata.entity][variable] = pd.Series(
            values,
            index=tables[metadata.entity].index,
        )

    for link in bundle.links:
        tables[link] = bundle.link(link).copy()
    weights = {
        entity: bundle.weights_for(entity) for entity in bundle.weighted_entities
    }
    return Frame(
        tables,
        bundle.schema,
        weights,
        strata=bundle.strata.copy(),
        mass_log=bundle.mass_log,
        metadata=bundle.metadata,
    )


def _as_str_tuple(values: Any) -> tuple[str, ...]:
    """Coerce a JSON list (or any iterable) into a tuple of strings."""
    if values is None:
        return ()
    return tuple(str(value) for value in values)
