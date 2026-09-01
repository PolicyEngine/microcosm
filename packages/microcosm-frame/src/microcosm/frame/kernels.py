"""Graph kernel wrapping a :class:`~microcosm.frame.rules.RulesEngine`.

The node declaration carries a serializable ``engine_ref``; the kernel instance
is bound to the corresponding adapter by the caller assembling its kernel
registry.  Binding the adapter outside the declaration keeps callables out of
node identity while allowing engine-free graphs to use a pure-Python adapter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.rules as frame_rules_module
import microcosm.frame.schema as frame_schema_module
from microcosm.frame.bundle import Frame
from microcosm.frame.rules import RulesEngine
from microcosm.frame.schema import EntitySchema
from microcosm.graph import (
    ROWS_ALL,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Numeric,
    Ownership,
    SeedSource,
    StructuralDelta,
    source_hash,
)

__all__ = ["RulesKernel", "SimulateRulesKernel"]


class SimulateRulesKernel(KernelBase):
    """Materialize declared variables through one bound rules-engine adapter.

    Args:
        engine_ref: Stable, non-empty reference carried in the node's parameters.
            The reference must uniquely identify the adapter and its configuration.
        engine: Adapter that performs the direct :meth:`RulesEngine.materialize`
            call.
        dependencies: Distribution names whose versions affect engine behavior.
            A PolicyEngine-US binding declares ``("policyengine-us",)``; a
            source-local test stub declares no distribution dependency.

    The wrapper reconstructs a complete :class:`~microcosm.frame.Frame` from the
    context's restricted entity tables, effective typed weights, and strata. It
    never fills engine inputs or reimplements a formula: absent inputs retain the
    direct adapter's ordinary default behavior.
    """

    ref = "simulate.rules@1"

    def __init__(
        self,
        engine_ref: str,
        engine: RulesEngine,
        dependencies: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(engine_ref, str) or not engine_ref:
            raise ValueError("engine_ref must be a non-empty string.")
        if not isinstance(engine, RulesEngine):
            raise TypeError(f"{engine!r} does not satisfy the RulesEngine protocol.")
        if not isinstance(dependencies, tuple) or any(
            not isinstance(name, str) or not name for name in dependencies
        ):
            raise TypeError("dependencies must be a tuple of non-empty strings.")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies must not repeat a distribution name.")

        self.engine_ref = engine_ref
        self._engine = engine
        self.capabilities = Capabilities(
            determinism=Determinism.DETERMINISTIC,
            numeric=Numeric.BITWISE,
            seed_source=SeedSource.NONE,
            structural=StructuralDelta.NONE,
            consumes_se=False,
            dependencies=dependencies,
        )

    def implementation_hash(self) -> str:
        """Hash both the wrapper and the bound adapter's implementation source."""

        return source_hash(
            type(self),
            type(self._engine),
            frame_bundle_module,
            frame_rules_module,
            frame_schema_module,
            dependencies=self.capabilities.dependencies,
        )

    @staticmethod
    def _tables(
        context: KernelContext, schema: EntitySchema
    ) -> dict[str, pd.DataFrame]:
        """Recover ID-only group tables omitted from the declared data slices.

        Kernel contexts expose structural person memberships alongside declared
        true-input columns.  A group with no true inputs may consequently have
        no slice of its own; Frame's partition invariant lets the adapter recover
        that group's sorted ID-only table exactly from the person memberships.
        """

        person_entity = schema.person_entity
        try:
            person = context.tables[person_entity]
        except KeyError as error:
            raise ValueError(
                "simulate.rules requires the engine's person table so group "
                "structure can be reconstructed."
            ) from error

        tables = {person_entity: person}
        for group in schema.group_entities:
            if group in context.tables:
                tables[group] = context.tables[group]
                continue
            membership = schema.membership_column(group)
            if membership not in person.columns:
                raise ValueError(
                    f"simulate.rules cannot reconstruct entity {group!r}: person "
                    f"table is missing structural membership {membership!r}."
                )
            ids = np.unique(person[membership].to_numpy(copy=False))
            tables[group] = pd.DataFrame({schema.id_column(group): ids})
        return tables

    def run(self, context: KernelContext) -> KernelResult:
        """Call ``RulesEngine.materialize`` and index its arrays by entity id."""

        engine_ref, variables, period = self._validated_params(context)
        schema = self._engine.entity_schema()

        tables = self._tables(context, schema)
        frame = Frame(
            tables=tables,
            schema=schema,
            weights=dict(context.weights),
            strata=context.strata,
        )

        metadata = {
            variable: self._engine.variable_metadata(variable) for variable in variables
        }
        expected_outputs = {
            (item.entity, variable) for variable, item in metadata.items()
        }
        declared_outputs = {
            (owned.entity, owned.column) for owned in context.node.outputs
        }
        if declared_outputs != expected_outputs:
            raise ValueError(
                "simulate.rules declared outputs do not match the requested engine "
                f"variables: declared={sorted(declared_outputs)!r}, "
                f"expected={sorted(expected_outputs)!r}."
            )
        invalid_scope = [
            f"{owned.entity}.{owned.column}"
            for owned in context.node.outputs
            if owned.rows != ROWS_ALL or owned.ownership is not Ownership.PRODUCED
        ]
        if invalid_scope:
            raise ValueError(
                "simulate.rules outputs must be produced on all entity rows; got "
                f"{invalid_scope}."
            )

        materialized = self._engine.materialize(frame, variables, period)
        columns: dict[tuple[str, str], pd.Series] = {}
        output_rows: list[tuple[str, str, int]] = []
        for variable in variables:
            if variable not in materialized:
                raise ValueError(
                    f"Rules engine did not return requested variable {variable!r}."
                )
            entity = metadata[variable].entity
            table = frame.table(entity)
            values = np.asarray(materialized[variable])
            expected_shape = (len(table),)
            if values.shape != expected_shape:
                raise ValueError(
                    f"Materialized variable {variable!r} has shape {values.shape} "
                    f"but entity {entity!r} has {len(table)} row(s)."
                )
            id_column = schema.entity_id_column(entity)
            ids = table[id_column].to_numpy(copy=False)
            columns[(entity, variable)] = pd.Series(
                values,
                index=pd.Index(ids, name=id_column),
                name=variable,
                copy=False,
            )
            output_rows.append((variable, entity, len(table)))

        return KernelResult(
            columns=columns,
            receipt={
                "engine_ref": engine_ref,
                "period": period,
                "variables": variables,
                "output_rows": tuple(output_rows),
            },
        )

    def _validated_params(
        self, context: KernelContext
    ) -> tuple[str, tuple[str, ...], int | str]:
        expected = {"engine_ref", "period", "variables"}
        actual = set(context.params)
        if actual != expected:
            raise ValueError(
                "simulate.rules parameters must be exactly "
                f"{sorted(expected)!r}; got {sorted(actual)!r}."
            )

        engine_ref = context.params["engine_ref"]
        if not isinstance(engine_ref, str) or not engine_ref:
            raise TypeError("simulate.rules engine_ref must be a non-empty string.")
        if engine_ref != self.engine_ref:
            raise ValueError(
                f"simulate.rules engine_ref {engine_ref!r} does not match the "
                f"bound engine {self.engine_ref!r}."
            )

        variables = context.params["variables"]
        if not isinstance(variables, tuple) or not variables:
            raise TypeError("simulate.rules variables must be a non-empty tuple.")
        if any(not isinstance(variable, str) or not variable for variable in variables):
            raise TypeError(
                "simulate.rules variables must contain only non-empty strings."
            )
        if len(set(variables)) != len(variables):
            raise ValueError("simulate.rules variables must not repeat a name.")

        period = context.params["period"]
        if isinstance(period, bool) or not isinstance(period, int | str):
            raise TypeError("simulate.rules period must be an int or non-empty string.")
        if isinstance(period, str) and not period:
            raise TypeError("simulate.rules period must be an int or non-empty string.")
        return engine_ref, variables, period


# Short spelling for callers that name the adapter rather than the graph stage.
RulesKernel = SimulateRulesKernel
