"""The pre-clone housing stage as a fresh-CPS-source graph recipe.

The recipe selects the real CPS source rows once through the registered pool
selection, derives the measured household/SPM leaves from their declared raw
tables, runs the existing independent ACS rent donor fit and recipient apply on
that projection, and merges the five declared leaves back into the unchanged
full pool by entity ID.

It is explicitly ``fresh_source``: selected CPS ``pre_subsidy_rent`` must be
absent or entirely null, and observed CPS rent is refused before any recipient
work. Native non-CPS rent and tenure survive untouched. The old direct
complete-incumbent pass-through in :func:`with_us_housing_inputs` is unchanged
and is not implemented here.

The outer gate attests source projection, exact ID merge and completion
integrity. The existing rent gate checks assignment integrity; the established
terminal scientific quality and release gates stay authoritative.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from importlib import import_module
from importlib.resources import files

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as fit_codec
from microcosm.fit import graph_legacy_apply_matrix, model_input, qrf, qrf_target
from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    KernelRole,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token

from . import _graph_acs_rent_artifacts as rent_codec
from . import _graph_housing_source_artifacts as codec
from . import graph_acs_rent as rent
from . import graph_context, housing_inputs
from .multispine_pool import (
    assert_multispine_source_merge_labels_supported,
    merge_multispine_housing_source_outputs,
    multispine_housing_boundary_columns,
    multispine_housing_implementation_contract,
    multispine_housing_output_family,
    multispine_housing_source_selection,
)
from .support_provenance import without_support_role_metadata

HOUSING_SOURCE_PREFIX = "early_preparation.housing"
HOUSING_SOURCE_POPULATION = f"{HOUSING_SOURCE_PREFIX}.source"
HOUSING_SOURCE_MEASURED = f"{HOUSING_SOURCE_PREFIX}.measured"
HOUSING_SOURCE_OUTPUTS = f"{HOUSING_SOURCE_PREFIX}.source_outputs"
HOUSING_SOURCE_MERGE_POPULATION = f"{HOUSING_SOURCE_PREFIX}.merge_boundary"
HOUSING_SOURCE_MERGE = f"{HOUSING_SOURCE_PREFIX}.merge"
HOUSING_SOURCE_ACCEPTED_CONTEXT = f"{HOUSING_SOURCE_PREFIX}.gate"
HOUSING_SOURCE_OUTER_PHASE = codec.HOUSING_PHASE
HOUSING_SOURCE_DEPENDENCIES = ("numpy", "pandas", "pyarrow")

# Whole-module closure of everything this recipe actually calls, matching the
# reviewed hours/CPS/prior boundaries. An edit anywhere in a listed module
# conservatively rekeys this stage, including the shared donor preparation.
_HOUSING_IMPLEMENTATION_MODULES = (
    "microcosm.build.us_runtime.graph_housing_source",
    "microcosm.build.us_runtime._graph_housing_source_artifacts",
    "microcosm.build.us_runtime.graph_acs_rent",
    "microcosm.build.us_runtime.graph_acs_rent_source",
    "microcosm.build.us_runtime._graph_acs_rent_artifacts",
    "microcosm.build.us_runtime.graph_context",
    "microcosm.build.us_runtime.housing_inputs",
    "microcosm.build.us_runtime.multispine_pool",
    "microcosm.build.us_runtime.operator_boundary",
    "microcosm.build.us_runtime.support_provenance",
    "microcosm.build.source_manifest",
    "microcosm.build.source_runtime",
    "microcosm.build.gates",
    "microcosm.fit",
    "microcosm.fit.graph_legacy_apply_matrix",
    "microcosm.fit.model_input",
    "microcosm.fit.qrf",
    "microcosm.fit.qrf_target",
    "microcosm.frame",
    "microcosm.frame.accounting",
    "microcosm.frame.bundle",
    "microcosm.frame.kernels",
    "microcosm.frame.materialize",
    "microcosm.frame.rules",
    "microcosm.frame.schema",
    "microcosm.frame.units",
    "microcosm.frame.weights",
    "microcosm.graph",
    "microcosm.graph.artifact_edges",
    "microcosm.graph.attachments",
    "microcosm.graph.availability",
    "microcosm.graph.canonical",
    "microcosm.graph.codecs",
    "microcosm.graph.decl",
    "microcosm.graph.errors",
    "microcosm.graph.executor",
    "microcosm.graph.explain",
    "microcosm.graph.kernel",
    "microcosm.graph.keys",
    "microcosm.graph.manifest",
    "microcosm.graph.population",
    "microcosm.graph.randomness",
    "microcosm.graph.serialize",
    "microcosm.graph.store",
    "microcosm.graph.view",
)

_CANONICAL_STRING = dtype_for_token("string")
_RENT_COLUMN = "pre_subsidy_rent"
_MEASURED_RAW = {
    "person": ("SPM_CAPHOUSESUB", "SPM_TENMORTSTATUS"),
    "household": ("H_TENURE",),
}


class _HousingKernel(KernelBase):
    """Bind the explicit module closure, resources and string runtime."""

    def implementation_hash(self) -> str:
        resource = files("microcosm.build.us").joinpath("source_stages.json")
        stage = housing_inputs.us_housing_inputs_stage_spec()
        return hashlib.sha256(
            canonical_json(
                {
                    "schema": "microcosm.us.housing-source-implementation.v1",
                    # The rent helper folds the ambient pandas string runtime and
                    # the declared dependency versions into one module digest.
                    "modules": rent_codec.encoding_implementation_hash(
                        type(self),
                        model_input,
                        graph_legacy_apply_matrix,
                        fit_codec,
                        qrf,
                        qrf_target,
                        *(
                            import_module(name)
                            for name in _HOUSING_IMPLEMENTATION_MODULES
                        ),
                        dependencies=HOUSING_SOURCE_DEPENDENCIES,
                    ),
                    "operator_contract": multispine_housing_implementation_contract(),
                    "output_family": {
                        entity: list(columns)
                        for entity, columns in sorted(
                            codec.housing_output_family().items()
                        )
                    },
                    "source_representation": {
                        f"{entity}.{column}": source
                        for entity, column, source, _merge in codec.SOURCE_COLUMNS
                    },
                    "merge_representation": {
                        f"{entity}.{column}": merge
                        for entity, column, merge in codec.merge_columns()
                    },
                    "stage_spec": {
                        "stage": stage.stage,
                        "survey": stage.survey,
                        "source": stage.source,
                        "grain": stage.grain,
                        "outputs": sorted(stage.outputs),
                        "rewrites": sorted(stage.rewrites),
                    },
                    "rent_predictors": list(housing_inputs.ACS_RENT_PREDICTORS),
                    "resources": {
                        "microcosm.build.us/source_stages.json": hashlib.sha256(
                            resource.read_bytes()
                        ).hexdigest()
                    },
                }
            )
        ).hexdigest()


def _assert_rewrite_declaration(
    context: KernelContext,
    document: Mapping[str, object],
    coordinates: Sequence[tuple[str, str]],
) -> None:
    """Own a carried cell as a rewrite, and a fresh cell as a non-rewrite.

    The executor supplies an incumbent only for a declared rewrite, so a caller
    whose column inventory omitted an existing leaf would silently replace its
    native cells with source-only values. The population's own context lists
    every carried column, so the mismatch is refused here instead of trusted
    from the declaration.
    """
    declared = {(item.entity, item.column): item for item in context.node.outputs}
    for entity, column in coordinates:
        owned = declared.get((entity, column))
        if owned is None:
            raise ValueError(
                f"US housing source does not own the declared leaf {entity}.{column}."
            )
        carried = column in document["entities"][entity]["columns"]
        if owned.rewrite != carried:
            state = "carries" if carried else "does not carry"
            raise ValueError(
                f"US housing source {entity}.{column} must declare "
                f"rewrite={carried!r}: this population {state} that incumbent."
            )


def _assert_declared_producers(
    context: KernelContext, expected: Mapping[str, str]
) -> None:
    """Require each typed alias to come from its authored producing node.

    Artifact edges cross population versions, so a payload digest alone cannot
    say which node published it. The declaration names the producer; checking it
    here refuses a swapped edge before any payload is trusted.
    """
    declared = {edge.name: edge.producer for edge in context.node.artifact_inputs}
    if any(declared.get(alias) != producer for alias, producer in expected.items()):
        raise ValueError(
            "US housing source requires its authored artifact producers; "
            f"declared={declared}, expected={dict(expected)}."
        )


def _phase(context: KernelContext, *, extra: frozenset[str] = frozenset()) -> None:
    if set(context.params) != {"phase", *extra} or context.params["phase"] != (
        HOUSING_SOURCE_OUTER_PHASE
    ):
        raise ValueError("US housing source requires its authored outer phase.")


def _mode(context: KernelContext) -> str:
    mode = context.params.get("mode")
    if mode != codec.FRESH_SOURCE_MODE:
        raise ValueError(f"Unsupported US housing source mode {mode!r}.")
    return mode


def _entity_view(
    table: pd.DataFrame, entity: str, declared: Mapping[str, object]
) -> pd.DataFrame:
    actual = graph_context._row_identity(table, entity)
    if any(declared[key] != value for key, value in actual.items()):
        raise ValueError(f"US housing source {entity} identity differs from context.")
    if set(table.columns) - set(declared["columns"]):
        raise ValueError(f"US housing source {entity} view includes unknown columns.")
    return table.loc[:, [name for name in declared["columns"] if name in table]]


def _context(
    context: KernelContext, *, alias: str = "frame_context", expected_type=None
):
    bound = codec.artifact(
        context, alias, expected_type or graph_context.US_FRAME_CONTEXT_TYPE
    )
    document = graph_context._decode(bound.payload)
    graph_context._mass_records(document["mass_log"])
    return document, bound


def _membership_counts(person: pd.DataFrame) -> dict[str, int]:
    return {
        group: int(person[US_SCHEMA.membership_column(group)].nunique())
        for group in US_SCHEMA.group_entities
    }


def _selection_descriptor() -> dict[str, str]:
    columns = multispine_housing_boundary_columns()
    return {
        "column": columns[0],
        "phase": codec.SOURCE_OPERATOR_PHASE,
        "clone_index_column": columns[1],
    }


def _merged_context(parent_payload: bytes) -> dict[str, object]:
    """Preserve the parent authority and append only the five owned leaf names."""
    document = graph_context._decode(parent_payload)
    for entity, column, _dtype in codec.merge_columns():
        declared = document["entities"][entity]["columns"]
        if column not in declared:
            declared.append(column)
    return document


def _selected_identities(
    person: pd.DataFrame, mask: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    selected = person.loc[mask]
    identities = {
        "person": {
            "full": codec.strict_ids(person["person_id"], label="person"),
            "selected": codec.strict_ids(selected["person_id"], label="person"),
        }
    }
    for group in US_SCHEMA.group_entities:
        column = US_SCHEMA.membership_column(group)
        codec.strict_ids(person[column], label=f"{group} membership", unique=False)
        identities[group] = {
            "full": np.unique(person[column].to_numpy(copy=False)),
            "selected": np.unique(selected[column].to_numpy(copy=False)),
        }
    return identities


class USHousingSourceBoundaryKernel(_HousingKernel):
    """FILTER the full pool to its registered CPS source rows, mass free."""

    ref = "us.production.housing_source_boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=HOUSING_SOURCE_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, extra=frozenset({"mode"}))
        mode = _mode(context)
        if (
            context.node.kernel != self.ref
            or context.node.outputs
            or context.node.sources
            or set(context.artifacts) != {"frame_context"}
            or context.node.structural is not StructuralDelta.FILTER
            or context.node.mass != "free"
            or set(context.tables) != {"person"}
        ):
            raise ValueError(
                "US housing source boundary requires its checked FILTER declaration."
            )
        document, bound = _context(context)
        person = _entity_view(
            context.tables["person"], "person", document["entities"]["person"]
        )
        counts = _membership_counts(person)
        for group, rows in counts.items():
            if document["entities"][group]["rows"] != rows:
                raise ValueError(
                    f"US housing source {group} context count differs from memberships."
                )
        mask, boundary = multispine_housing_source_selection(
            person, metadata=document["metadata"]
        )
        selection = mask.to_numpy(dtype=bool, copy=True)
        if not selection.any():
            # The registered mask already refuses this; keep the local guard so
            # Frame.select can never be handed an empty selection.
            raise ValueError("US housing source found no CPS-evidenced person rows.")
        # Fresh-source scope: refuse observed CPS rent before any recipient work.
        if _RENT_COLUMN in document["entities"]["person"]["columns"] and (
            _RENT_COLUMN not in person
        ):
            raise ValueError(
                "US housing source boundary must declare the available "
                "pre_subsidy_rent incumbent so its fresh-source scope is checked."
            )
        if _RENT_COLUMN in person:
            incumbent = person.loc[selection, _RENT_COLUMN]
            if str(incumbent.dtype) != "float64":
                raise ValueError(
                    "US housing source rent incumbent must be unavailable float64."
                )
            if incumbent.notna().any():
                raise ValueError(
                    "US housing source is fresh_source only: the selected CPS rows "
                    "already carry observed pre_subsidy_rent. The direct "
                    "complete-incumbent pass-through remains the owner of that case."
                )
        identities = _selected_identities(person, selection)
        for entity, values in identities.items():
            full = document["entities"][entity]
            if full["rows"] != len(values["full"]):
                raise ValueError(
                    f"US housing source {entity} parent identity differs from context."
                )
            if not np.isin(values["selected"], values["full"]).all():
                raise ValueError(
                    f"US housing source selected {entity} IDs leave the parent pool."
                )
        projection = codec.encode_projection(
            parent_context=bound,
            selection=_selection_descriptor(),
            entities=identities,
            person_index=person.index,
            weight_sources=document["weight_sources"],
            mode=mode,
        )
        projected = dict(document)
        projected["metadata"] = {}
        projected["mass_log"] = []
        projected["entities"] = {
            entity: {
                "columns": list(document["entities"][entity]["columns"]),
                **graph_context._row_identity(
                    pd.DataFrame(
                        {
                            US_SCHEMA.entity_id_column(entity): pd.Series(
                                identities[entity]["selected"], dtype="int64"
                            )
                        }
                    ),
                    entity,
                ),
            }
            for entity in US_SCHEMA.entities
        }
        return KernelResult(
            keep=pd.Series(
                selection,
                index=pd.Index(person["person_id"], name="person_id"),
                dtype=bool,
            ),
            artifacts={
                "frame_context": canonical_json(projected),
                "projection": projection,
            },
            receipt={
                "phase": HOUSING_SOURCE_OUTER_PHASE,
                "mode": mode,
                "source_boundary": boundary,
                "selected_rows": {
                    entity: int(len(values["selected"]))
                    for entity, values in identities.items()
                },
            },
        )


class USHousingMeasuredKernel(_HousingKernel):
    """Derive the measured household/SPM leaves on the source projection."""

    ref = "us.production.housing_measured@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=HOUSING_SOURCE_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, extra=frozenset({"mode"}))
        mode = _mode(context)
        if (
            context.node.kernel != self.ref
            or context.node.sources
            or context.node.structural is not StructuralDelta.NONE
            or set(context.artifacts) != {"frame_context", "projection"}
            or set(context.tables) != {"person", "household", "spm_unit"}
        ):
            raise ValueError(
                "US housing measured requires its exact declared source views."
            )
        _assert_declared_producers(
            context,
            {
                "frame_context": HOUSING_SOURCE_POPULATION,
                "projection": HOUSING_SOURCE_POPULATION,
            },
        )
        document, _bound = _context(context)
        projection = codec.decode_projection(
            codec.artifact(
                context, "projection", codec.HOUSING_SOURCE_PROJECTION_TYPE
            ).payload
        )
        if projection["mode"] != mode:
            raise ValueError("US housing measured mode differs from its projection.")
        tables = {
            entity: _entity_view(table, entity, document["entities"][entity])
            for entity, table in context.tables.items()
        }
        for entity, table in tables.items():
            actual = graph_context._row_identity(table, entity)
            declared = projection["entities"][entity]["selected"]
            if any(declared[key] != value for key, value in actual.items()):
                raise ValueError(
                    f"US housing measured {entity} rows differ from the projection."
                )
        for entity, required in _MEASURED_RAW.items():
            missing = [name for name in required if name not in tables[entity]]
            if missing:
                raise ValueError(
                    f"US housing measured declaration omits raw {entity} input(s) "
                    f"{missing}."
                )
        expected = {
            (entity, column, source)
            for entity, column, source, _merge in codec.SOURCE_COLUMNS
            if entity != "person"
        }
        if {
            (value.entity, value.column, value.dtype) for value in context.node.outputs
        } != expected:
            raise ValueError(
                "US housing measured must own exactly its four declared leaves."
            )
        _assert_rewrite_declaration(
            context,
            document,
            sorted((entity, column) for entity, column, _d in expected),
        )
        # Role columns can ride along on a FILTER; the ephemeral scientific view
        # removes them, exactly as the registered source projection does.
        views = {
            entity: without_support_role_metadata(table, entity=entity)
            for entity, table in tables.items()
        }
        household, spm_unit = housing_inputs.derive_us_housing_tables(
            views["person"], views["household"], views["spm_unit"]
        )
        produced = {"household": household, "spm_unit": spm_unit}
        columns: dict[tuple[str, str], pd.Series] = {}
        for owned in context.node.outputs:
            table = produced[owned.entity]
            values = self._canonical(
                table[owned.column],
                incumbent=tables[owned.entity].get(owned.column),
                owned=owned,
            )
            columns[(owned.entity, owned.column)] = pd.Series(
                values.array,
                index=pd.Index(
                    table[US_SCHEMA.entity_id_column(owned.entity)],
                    name=US_SCHEMA.entity_id_column(owned.entity),
                ),
            )
        for entity, names in multispine_housing_output_family().items():
            declared = document["entities"][entity]["columns"]
            for name in sorted(names):
                if (entity, name) in columns and name not in declared:
                    declared.append(name)
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": canonical_json(document)},
            receipt={
                "phase": HOUSING_SOURCE_OUTER_PHASE,
                "mode": mode,
                "measured_rows": {
                    entity: int(len(table)) for entity, table in produced.items()
                },
                "scope": "measured_household_and_spm_leaves",
            },
        )

    @staticmethod
    def _canonical(
        values: pd.Series, *, incumbent: pd.Series | None, owned: Owned
    ) -> pd.Series:
        """Convert derived values to the declared representation, exactly."""
        label = f"{owned.entity}.{owned.column}"
        if owned.dtype == "bool":
            if not pd.api.types.is_bool_dtype(values.dtype) or values.isna().any():
                raise ValueError(
                    f"US housing measured {label} requires physical booleans."
                )
            return values.astype(bool)
        if owned.dtype != "string":
            raise ValueError(f"US housing measured {label} has an unsupported dtype.")
        if incumbent is not None and (
            not isinstance(incumbent.dtype, pd.StringDtype)
            or incumbent.dtype != _CANONICAL_STRING
            or incumbent.dtype.storage != _CANONICAL_STRING.storage
        ):
            raise ValueError(
                f"US housing measured {label} requires the canonical string "
                "incumbent representation; refusing to widen the declared scope."
            )
        raw = values.to_numpy(dtype=object, copy=False)
        if any(value is None or value is pd.NA for value in raw) or any(
            not isinstance(value, str) for value in raw
        ):
            raise ValueError(
                f"US housing measured {label} derived a non-string or null value."
            )
        converted = pd.Series(raw, dtype=_CANONICAL_STRING)
        if converted.isna().any() or not np.array_equal(
            converted.to_numpy(dtype=object), raw
        ):
            raise ValueError(
                f"US housing measured {label} lost values in its string conversion."
            )
        return converted


class USHousingSourceOutputsKernel(_HousingKernel):
    """Export the five completed source leaves as immutable typed bytes."""

    ref = "us.production.housing_source_outputs@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=HOUSING_SOURCE_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, extra=frozenset({"mode"}))
        mode = _mode(context)
        if (
            context.node.kernel != self.ref
            or context.node.outputs
            or context.node.sources
            or context.node.structural is not StructuralDelta.NONE
            or set(context.artifacts) != {"frame_context", "projection", "validation"}
            or context.node.artifact_outputs
            != (ArtifactOutput("source_outputs", codec.HOUSING_SOURCE_OUTPUTS_TYPE),)
            or set(context.tables) != {"person", "household", "spm_unit"}
        ):
            raise ValueError(
                "US housing source outputs requires its exact export declaration."
            )
        string_runtime = codec.string_runtime_identity()
        document, bound = _context(context)
        projection_value = codec.artifact(
            context, "projection", codec.HOUSING_SOURCE_PROJECTION_TYPE
        )
        projection = codec.decode_projection(projection_value.payload)
        validation_value = codec.artifact(
            context, "validation", rent_codec.ACS_RENT_VALIDATION_TYPE
        )
        if projection["mode"] != mode:
            raise ValueError("US housing outputs mode differs from its projection.")
        _assert_declared_producers(
            context,
            {
                "frame_context": rent.RENT_OUTPUT_CONTEXT,
                "projection": HOUSING_SOURCE_POPULATION,
                "validation": f"{rent.RENT_PREFIX}.validation",
            },
        )
        tables = {}
        for entity, expected in codec.source_tables().items():
            table = _entity_view(
                context.tables[entity], entity, document["entities"][entity]
            )
            declared = projection["entities"][entity]["selected"]
            actual = graph_context._row_identity(table, entity)
            if any(declared[key] != value for key, value in actual.items()):
                raise ValueError(
                    f"US housing outputs {entity} rows differ from the projection."
                )
            missing = [name for name, _dtype in expected if name not in table]
            if missing:
                raise ValueError(f"US housing outputs {entity} view omits {missing}.")
            tables[entity] = table.loc[
                :, [US_SCHEMA.entity_id_column(entity), *(name for name, _ in expected)]
            ]
        payload = codec.encode_source_outputs(
            tables,
            projection=projection_value,
            frame_context=bound,
            validation=validation_value,
            mode=mode,
        )
        codec.assert_string_runtime(string_runtime)
        return KernelResult(
            artifacts={"source_outputs": payload},
            receipt={
                "phase": HOUSING_SOURCE_OUTER_PHASE,
                "mode": mode,
                "source_outputs_sha256": codec.sha(payload),
                "validation_sha256": codec.sha(validation_value.payload),
                "rows": {entity: int(len(table)) for entity, table in tables.items()},
                "string_runtime": string_runtime,
            },
        )


class USHousingMergeBoundaryKernel(_HousingKernel):
    """Open an isolated full-pool version once the source export completed."""

    ref = "us.production.housing_merge_boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=HOUSING_SOURCE_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, extra=frozenset({"mode"}))
        mode = _mode(context)
        if (
            context.node.kernel != self.ref
            or context.node.outputs
            or context.node.sources
            or set(context.artifacts)
            != {"frame_context", "projection", "source_outputs"}
            or context.node.structural is not StructuralDelta.FILTER
            or context.node.mass != "conserve"
            or set(context.tables) != {"person"}
        ):
            raise ValueError(
                "US housing merge boundary requires its all-row declaration."
            )
        _assert_declared_producers(
            context,
            {
                "projection": HOUSING_SOURCE_POPULATION,
                "source_outputs": HOUSING_SOURCE_OUTPUTS,
            },
        )
        document, bound = _context(context)
        person = _entity_view(
            context.tables["person"], "person", document["entities"]["person"]
        )
        for group, rows in _membership_counts(person).items():
            if document["entities"][group]["rows"] != rows:
                raise ValueError(
                    f"US housing merge boundary would prune unreferenced {group} rows."
                )
        projection_value = codec.artifact(
            context, "projection", codec.HOUSING_SOURCE_PROJECTION_TYPE
        )
        projection = codec.decode_projection(projection_value.payload)
        outputs_value = codec.artifact(
            context, "source_outputs", codec.HOUSING_SOURCE_OUTPUTS_TYPE
        )
        outputs = codec.decode_source_outputs(outputs_value.payload)
        bindings = codec.binding(projection_value)
        if (
            outputs.header["projection_sha256"] != bindings["sha256"]
            or outputs.header["projection_producer_key"] != bindings["producer_key"]
        ):
            raise ValueError(
                "US housing merge boundary output/projection binding is stale."
            )
        if (
            projection["parent_context_sha256"] != codec.sha(bound.payload)
            or projection["parent_context_producer_key"] != bound.producer_key
            or projection["mode"] != mode
        ):
            raise ValueError(
                "US housing merge boundary projection binds a different full pool."
            )
        return KernelResult(
            keep=pd.Series(
                True, index=pd.Index(person["person_id"], name="person_id"), dtype=bool
            ),
            artifacts={"frame_context": bound.payload},
            receipt={
                "phase": HOUSING_SOURCE_OUTER_PHASE,
                "mode": mode,
                "scope": "full_pool_merge_boundary",
                "source_outputs_sha256": codec.sha(outputs_value.payload),
            },
        )


class USHousingMergeKernel(_HousingKernel):
    """Merge the five declared leaves into the unchanged full pool by ID."""

    ref = "us.production.housing_merge@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=HOUSING_SOURCE_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, extra=frozenset({"mode"}))
        mode = _mode(context)
        if (
            context.node.kernel != self.ref
            or context.node.sources
            or context.node.structural is not StructuralDelta.NONE
            or set(context.artifacts)
            != {"frame_context", "projection", "source_outputs"}
            or context.node.artifact_outputs
            != (
                ArtifactOutput("frame_context", codec.HOUSING_PENDING_CONTEXT_TYPE),
                ArtifactOutput("merge_evidence", codec.HOUSING_MERGE_EVIDENCE_TYPE),
            )
            or set(context.tables) != {"person", "household", "spm_unit"}
        ):
            raise ValueError("US housing merge requires its exact declaration.")
        if {
            (value.entity, value.column, value.dtype) for value in context.node.outputs
        } != set(codec.merge_columns()):
            raise ValueError(
                "US housing merge must own exactly its five full-pool leaves."
            )
        _assert_declared_producers(
            context,
            {
                "frame_context": HOUSING_SOURCE_MERGE_POPULATION,
                "projection": HOUSING_SOURCE_POPULATION,
                "source_outputs": HOUSING_SOURCE_OUTPUTS,
            },
        )
        document, bound = _context(context)
        _assert_rewrite_declaration(
            context,
            document,
            [(entity, column) for entity, column, _dtype in codec.merge_columns()],
        )
        projection_value = codec.artifact(
            context, "projection", codec.HOUSING_SOURCE_PROJECTION_TYPE
        )
        projection = codec.decode_projection(projection_value.payload)
        outputs_value = codec.artifact(
            context, "source_outputs", codec.HOUSING_SOURCE_OUTPUTS_TYPE
        )
        outputs = codec.decode_source_outputs(outputs_value.payload)
        bindings = codec.binding(projection_value)
        # The merge boundary republishes the full-pool context bytes verbatim and
        # is the node that authenticated the parent producer key, so this node
        # binds the payload digest and its own declared producer.
        if (
            outputs.header["projection_sha256"] != bindings["sha256"]
            or outputs.header["projection_producer_key"] != bindings["producer_key"]
            or projection["parent_context_sha256"] != codec.sha(bound.payload)
            or projection["mode"] != mode
            or outputs.header["mode"] != mode
        ):
            raise ValueError("US housing merge received stale producer bindings.")

        tables = {
            entity: _entity_view(table, entity, document["entities"][entity])
            for entity, table in context.tables.items()
        }
        person = tables["person"]
        mask, boundary = multispine_housing_source_selection(
            person, metadata=document["metadata"]
        )
        selection = mask.to_numpy(dtype=bool, copy=True)
        identities = _selected_identities(person, selection)
        for entity, values in identities.items():
            declared = projection["entities"][entity]
            for key, ids in (
                ("full", values["full"]),
                ("selected", values["selected"]),
            ):
                actual = graph_context._row_identity(
                    pd.DataFrame(
                        {
                            US_SCHEMA.entity_id_column(entity): pd.Series(
                                ids, dtype="int64"
                            )
                        }
                    ),
                    entity,
                )
                if any(declared[key][name] != value for name, value in actual.items()):
                    raise ValueError(
                        f"US housing merge recomputed {entity} {key} identity differs "
                        "from the bound projection."
                    )
        for entity in codec.SOURCE_ENTITIES:
            np.testing.assert_array_equal(
                outputs.tables[entity][US_SCHEMA.entity_id_column(entity)].to_numpy(),
                identities[entity]["selected"],
                err_msg=f"US housing merge {entity} source IDs differ from selection.",
            )

        pool_rows = {
            entity: int(value["rows"]) for entity, value in document["entities"].items()
        }
        selected_rows = {
            entity: int(len(identities[entity]["selected"]))
            for entity in US_SCHEMA.entities
        }

        def native_mask(entity: str) -> np.ndarray:
            return ~(
                tables[entity][US_SCHEMA.entity_id_column(entity)]
                .isin(identities[entity]["selected"])
                .to_numpy()
            )

        # Digest the native cells *before* the merge as well as after, so the
        # gate can attest preservation from the evidence itself rather than
        # inheriting this kernel's word for it.
        incumbent_digests: dict[tuple[str, str], dict[str, object]] = {}
        for entity, column, _dtype in codec.merge_columns():
            if column in tables[entity]:
                incumbent_digests[(entity, column)] = codec.cell_digest(
                    tables[entity].loc[native_mask(entity), column],
                    entity=entity,
                    column=column,
                )
        merged, merged_rows, receipt = merge_multispine_housing_source_outputs(
            {entity: table.copy(deep=True) for entity, table in tables.items()},
            {
                entity: outputs.tables[entity].copy(deep=True)
                for entity in codec.SOURCE_ENTITIES
            },
            pool_rows=pool_rows,
            selected_rows=selected_rows,
            evidence_rows=boundary["cps_person_rows"],
        )
        columns: dict[tuple[str, str], pd.Series] = {}
        merged_cells = []
        native_cells = []
        incumbent_cells = []
        for entity, column, dtype in codec.merge_columns():
            values = merged[entity][column]
            if str(values.dtype) != dtype:
                raise ValueError(
                    f"US housing merge produced {entity}.{column} as {values.dtype!s}, "
                    f"not the declared {dtype!r}."
                )
            if (entity, column) in incumbent_digests:
                preserved = codec.cell_digest(
                    values.loc[native_mask(entity)], entity=entity, column=column
                )
                if preserved != incumbent_digests[(entity, column)]:
                    raise ValueError(
                        f"US housing merge changed native {entity}.{column} cells."
                    )
                incumbent_cells.append(incumbent_digests[(entity, column)])
                native_cells.append(preserved)
            merged_cells.append(codec.cell_digest(values, entity=entity, column=column))
            columns[(entity, column)] = pd.Series(
                values.array,
                index=pd.Index(
                    merged[entity][US_SCHEMA.entity_id_column(entity)],
                    name=US_SCHEMA.entity_id_column(entity),
                ),
            )
        pending = canonical_json(_merged_context(canonical_json(document)))
        evidence = codec.encode_merge_evidence(
            projection=projection_value,
            source_outputs=outputs_value,
            pending_context=pending,
            merged_rows=merged_rows,
            selected_rows=selected_rows,
            pool_rows=pool_rows,
            merged_cells=merged_cells,
            native_cells=native_cells,
            incumbent_cells=incumbent_cells,
            source_preparation=receipt,
            mode=mode,
        )
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": pending, "merge_evidence": evidence},
            receipt={
                "phase": HOUSING_SOURCE_OUTER_PHASE,
                "mode": mode,
                "source_preparation": receipt,
                "acceptance": "pending_completion_gate",
            },
        )


class USHousingSourceGateKernel(_HousingKernel):
    """Attest projection, exact ID merge and completion; not a new signal gate."""

    ref = "us.production.housing_source_gate@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        role=KernelRole.GATE,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=HOUSING_SOURCE_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, extra=frozenset({"mode"}))
        mode = _mode(context)
        if (
            context.node.kernel != self.ref
            or context.node.outputs
            or context.node.sources
            or set(context.artifacts)
            != {
                "frame_context",
                "parent_frame_context",
                "source_frame_context",
                "merge_evidence",
                "projection",
                "source_outputs",
                "validation",
            }
            or context.node.artifact_outputs
            != (ArtifactOutput("frame_context", graph_context.US_FRAME_CONTEXT_TYPE),)
            or set(context.tables) != {"person", "household", "spm_unit"}
        ):
            raise ValueError("US housing gate requires its exact declaration.")
        _assert_declared_producers(
            context,
            {
                "frame_context": HOUSING_SOURCE_MERGE,
                "source_frame_context": rent.RENT_OUTPUT_CONTEXT,
                "merge_evidence": HOUSING_SOURCE_MERGE,
                "projection": HOUSING_SOURCE_POPULATION,
                "source_outputs": HOUSING_SOURCE_OUTPUTS,
                "validation": f"{rent.RENT_PREFIX}.validation",
            },
        )
        pending = codec.artifact(
            context, "frame_context", codec.HOUSING_PENDING_CONTEXT_TYPE
        )
        evidence_value = codec.artifact(
            context, "merge_evidence", codec.HOUSING_MERGE_EVIDENCE_TYPE
        )
        if pending.producer_key != evidence_value.producer_key:
            raise ValueError(
                "US housing gate evidence and pending context require one producer."
            )
        evidence = codec.decode_merge_evidence(evidence_value.payload)
        if evidence["pending_context_sha256"] != codec.sha(pending.payload):
            raise ValueError(
                "US housing gate evidence belongs to a different pending context."
            )
        document = graph_context._decode(pending.payload)
        graph_context._mass_records(document["mass_log"])
        parent_document, parent_value = _context(context, alias="parent_frame_context")
        source_document, source_value = _context(context, alias="source_frame_context")
        if pending.payload != canonical_json(_merged_context(parent_value.payload)):
            raise ValueError(
                "US housing gate pending context does not preserve its actual parent "
                "authority and append exactly the owned housing columns."
            )
        projection_value = codec.artifact(
            context, "projection", codec.HOUSING_SOURCE_PROJECTION_TYPE
        )
        projection = codec.decode_projection(projection_value.payload)
        outputs_value = codec.artifact(
            context, "source_outputs", codec.HOUSING_SOURCE_OUTPUTS_TYPE
        )
        outputs = codec.decode_source_outputs(outputs_value.payload)
        validation_value = codec.artifact(
            context, "validation", rent_codec.ACS_RENT_VALIDATION_TYPE
        )
        projection_binding = codec.binding(projection_value)
        outputs_binding = codec.binding(outputs_value)
        validation_binding = codec.binding(validation_value)
        parent_binding = codec.binding(parent_value)
        source_binding = codec.binding(source_value)
        if (
            projection["parent_context_sha256"] != parent_binding["sha256"]
            or projection["parent_context_producer_key"]
            != parent_binding["producer_key"]
            or outputs.header["frame_context_sha256"] != source_binding["sha256"]
            or outputs.header["frame_context_producer_key"]
            != source_binding["producer_key"]
            or evidence["projection_sha256"] != projection_binding["sha256"]
            or evidence["projection_producer_key"] != projection_binding["producer_key"]
            or evidence["source_outputs_sha256"] != outputs_binding["sha256"]
            or evidence["source_outputs_producer_key"]
            != outputs_binding["producer_key"]
            or outputs.header["projection_sha256"] != projection_binding["sha256"]
            or outputs.header["projection_producer_key"]
            != projection_binding["producer_key"]
            or outputs.header["validation_sha256"] != validation_binding["sha256"]
            or outputs.header["validation_producer_key"]
            != validation_binding["producer_key"]
            or evidence["mode"] != mode
            or projection["mode"] != mode
            or outputs.header["mode"] != mode
        ):
            raise ValueError("US housing gate producer bindings are stale or swapped.")
        if (
            projection["selection"] != _selection_descriptor()
            or projection["weight_sources"] != parent_document["weight_sources"]
        ):
            raise ValueError(
                "US housing gate projection selection or weight authority differs "
                "from the registered selection and actual parent context."
            )
        for entity, identities in projection["entities"].items():
            for phase, authority in (
                ("full", parent_document),
                ("selected", source_document),
            ):
                if any(
                    authority["entities"][entity][name] != value
                    for name, value in identities[phase].items()
                ):
                    raise ValueError(
                        f"US housing gate projection {entity} {phase} identity "
                        "differs from its actual context authority."
                    )
        expected_validation = codec.encode_json(
            {
                "schema_version": 1,
                "passed": True,
                "failure": None,
                "bindings": codec.decode_json(validation_value.payload)["bindings"],
            }
        )
        if validation_value.payload != expected_validation:
            raise ValueError(
                "US housing gate requires the completed rent assignment-integrity gate."
            )

        tables = {
            entity: _entity_view(table, entity, document["entities"][entity])
            for entity, table in context.tables.items()
        }
        person = tables["person"]
        if projection["person_index"] != codec.person_index_identity(person.index):
            raise ValueError(
                "US housing gate projection person index differs from the actual pool."
            )
        mask, boundary = multispine_housing_source_selection(
            person, metadata=document["metadata"]
        )
        selection = mask.to_numpy(dtype=bool, copy=True)
        identities = _selected_identities(person, selection)
        pool_rows = {
            entity: int(value["rows"]) for entity, value in document["entities"].items()
        }
        selected_rows = {
            entity: int(len(identities[entity]["selected"]))
            for entity in US_SCHEMA.entities
        }
        if dict(evidence["pool_rows"]) != pool_rows or (
            dict(evidence["selected_rows"]) != selected_rows
        ):
            raise ValueError("US housing gate row counts differ from the merged pool.")
        expected_merged = {
            entity: selected_rows[entity] for entity in codec.SOURCE_ENTITIES
        }
        if dict(evidence["merged_rows"]) != expected_merged:
            raise ValueError(
                "US housing gate merged row counts differ from the selection."
            )
        receipt = evidence["source_preparation"]
        if receipt["cps_source_evidence"] != {
            "column": "PERIDNUM",
            "person_rows": boundary["cps_person_rows"],
        }:
            raise ValueError("US housing gate evidence differs from actual selection.")

        merged_cells = {
            (cell["entity"], cell["column"]): cell for cell in evidence["merged_cells"]
        }
        native_cells = {
            (cell["entity"], cell["column"]): cell for cell in evidence["native_cells"]
        }
        incumbent_cells = {
            (cell["entity"], cell["column"]): cell
            for cell in evidence["incumbent_cells"]
        }
        expected_native = {
            (entity, column)
            for entity, column, _dtype in codec.merge_columns()
            if column in parent_document["entities"][entity]["columns"]
        }
        if (
            set(incumbent_cells) != expected_native
            or set(native_cells) != expected_native
        ):
            raise ValueError(
                "US housing gate requires before/after native evidence for every "
                "actual parent incumbent."
            )
        for entity, column, dtype in codec.merge_columns():
            values = tables[entity][column]
            if str(values.dtype) != dtype:
                raise ValueError(
                    f"US housing gate found {entity}.{column} as {values.dtype!s}."
                )
            actual = codec.cell_digest(values, entity=entity, column=column)
            if actual != merged_cells.get((entity, column)):
                raise ValueError(
                    f"US housing gate final {entity}.{column} differs from evidence."
                )
            selected = tables[entity][US_SCHEMA.entity_id_column(entity)].isin(
                identities[entity]["selected"]
            )
            native = native_cells.get((entity, column))
            if native is not None:
                observed = codec.cell_digest(
                    values.loc[~selected.to_numpy()], entity=entity, column=column
                )
                # The evidence carries the native cells as the merge saw them
                # before it wrote, and as they stood after. Requiring both to
                # equal what this gate independently recomputes attests
                # preservation instead of trusting the merge kernel's own check.
                if observed != native or incumbent_cells[(entity, column)] != native:
                    raise ValueError(
                        f"US housing gate native {entity}.{column} cells changed."
                    )
            if entity in codec.SOURCE_ENTITIES:
                source_values = outputs.tables[entity][column]
                pool_values = values.loc[selected.to_numpy()]
                if len(pool_values) != len(source_values):
                    raise ValueError(
                        f"US housing gate {entity}.{column} selection width differs."
                    )
                if not np.array_equal(
                    pool_values.to_numpy(dtype=object),
                    source_values.to_numpy(dtype=object),
                ):
                    raise ValueError(
                        f"US housing gate {entity}.{column} differs from its source "
                        "output artifact."
                    )
        accepted = graph_context._decode(pending.payload)
        return KernelResult(
            artifacts={"frame_context": canonical_json(accepted)},
            receipt={
                "outcome": "pass",
                "phase": HOUSING_SOURCE_OUTER_PHASE,
                "mode": mode,
                "scope": codec.HOUSING_GATE_SCOPE,
                "source_preparation": receipt,
                "rent_validation_sha256": validation_binding["sha256"],
                "merge_evidence_sha256": codec.sha(evidence_value.payload),
            },
        )


def _inventory(columns: Sequence[Owned]) -> dict[tuple[str, str], Owned]:
    inventory = {(item.entity, item.column): item for item in columns}
    if len(inventory) != len(columns):
        raise ValueError("US housing source declarations repeat coordinates.")
    return inventory


def us_housing_source_nodes(
    columns: Sequence[Owned],
    *,
    base: str,
    context_producer: str,
    donor_descriptor: bytes,
    seed: int,
    n_estimators: int = housing_inputs._DEFAULT_N_ESTIMATORS,
    mode: str = codec.FRESH_SOURCE_MODE,
) -> tuple[Node, ...]:
    """Declare the fresh-source housing stage over the caller's full pool.

    ``base`` is the assembled pre-clone pool version and ``context_producer``
    its accepted typed context, which must already carry the real relationship
    stage. The donor descriptor is the existing authenticated rent preflight.
    """
    if mode != codec.FRESH_SOURCE_MODE:
        raise ValueError(
            f"Unsupported US housing source mode {mode!r}; this recipe implements "
            "fresh_source only and does not claim the complete-incumbent path."
        )
    inventory = _inventory(columns)
    # Reads the live registered roster and refuses if it no longer matches the
    # five leaves this recipe declares.
    codec.housing_output_family()
    boundary_columns = multispine_housing_boundary_columns()
    for column in boundary_columns:
        if ("person", column) not in inventory:
            raise ValueError(f"US housing source boundary input {column!r} is absent.")
    for entity, required in _MEASURED_RAW.items():
        for column in required:
            if (entity, column) not in inventory:
                raise ValueError(
                    f"US housing source requires declared raw {entity}.{column}."
                )
    rent_incumbent = inventory.get(("person", _RENT_COLUMN))
    if rent_incumbent is not None and rent_incumbent.dtype != "float64":
        raise ValueError("US housing source rent incumbent must be float64.")

    measured_outputs = []
    for entity, column, source_dtype, _merge in codec.SOURCE_COLUMNS:
        if entity == "person":
            continue
        incumbent = inventory.get((entity, column))
        if source_dtype == "string":
            if incumbent is None or incumbent.dtype != "string":
                raise ValueError(
                    f"US housing source requires a canonical string incumbent for "
                    f"{entity}.{column}; refusing an absent or incompatible "
                    "representation."
                )
        elif incumbent is not None:
            raise ValueError(
                "US housing fresh_source requires assistance leaves to be absent: "
                f"{entity}.{column} is already carried. Source bool and full-pool "
                "nullable boolean representations cannot both rewrite one "
                "carried dtype in this recipe."
            )
        measured_outputs.append(
            Owned(entity, column, source_dtype, rewrite=incumbent is not None)
        )

    mode_params = {"phase": HOUSING_SOURCE_OUTER_PHASE, "mode": mode}
    source_context = (
        ArtifactOutput("frame_context", graph_context.US_FRAME_CONTEXT_TYPE),
    )
    projection_output = (
        ArtifactOutput("projection", codec.HOUSING_SOURCE_PROJECTION_TYPE),
    )
    parent_context_edge = ArtifactInput(
        "frame_context",
        context_producer,
        "frame_context",
        graph_context.US_FRAME_CONTEXT_TYPE,
    )
    projection_edge = ArtifactInput(
        "projection",
        HOUSING_SOURCE_POPULATION,
        "projection",
        codec.HOUSING_SOURCE_PROJECTION_TYPE,
    )
    boundary_slice_columns = tuple(
        dict.fromkeys((*boundary_columns, *((_RENT_COLUMN,) if rent_incumbent else ())))
    )
    source = Node(
        HOUSING_SOURCE_POPULATION,
        USHousingSourceBoundaryKernel.ref,
        base=base,
        structural=StructuralDelta.FILTER,
        mass="free",
        inputs=(Slice("person", boundary_slice_columns),),
        params=mode_params,
        artifact_inputs=(parent_context_edge,),
        artifact_outputs=(*source_context, *projection_output),
    )
    measured_inputs = []
    for entity in ("person", "household", "spm_unit"):
        names = list(_MEASURED_RAW.get(entity, ()))
        if entity == "person":
            names.extend(boundary_columns)
        names = [name for name in dict.fromkeys(names) if (entity, name) in inventory]
        if names:
            measured_inputs.append(Slice(entity, tuple(names)))
    measured = Node(
        HOUSING_SOURCE_MEASURED,
        USHousingMeasuredKernel.ref,
        population=HOUSING_SOURCE_POPULATION,
        inputs=tuple(measured_inputs),
        outputs=tuple(measured_outputs),
        params=mode_params,
        artifact_inputs=(
            ArtifactInput(
                "frame_context",
                HOUSING_SOURCE_POPULATION,
                "frame_context",
                graph_context.US_FRAME_CONTEXT_TYPE,
            ),
            projection_edge,
        ),
        artifact_outputs=source_context,
    )
    source_inventory = dict(inventory)
    for owned in measured_outputs:
        source_inventory[(owned.entity, owned.column)] = owned
    rent_nodes = rent.us_acs_rent_nodes(
        tuple(source_inventory.values()),
        base=HOUSING_SOURCE_POPULATION,
        context_producer=HOUSING_SOURCE_MEASURED,
        donor_descriptor=donor_descriptor,
        seed=seed,
        n_estimators=n_estimators,
    )
    validation_edge = ArtifactInput(
        "validation",
        f"{rent.RENT_PREFIX}.validation",
        "validation",
        rent_codec.ACS_RENT_VALIDATION_TYPE,
    )
    export_slices = tuple(
        Slice(entity, tuple(name for name, _dtype in expected))
        for entity, expected in sorted(codec.source_tables().items())
    )
    source_outputs = Node(
        HOUSING_SOURCE_OUTPUTS,
        USHousingSourceOutputsKernel.ref,
        population=rent.RENT_OUTPUT_POPULATION,
        inputs=export_slices,
        params=mode_params,
        artifact_inputs=(
            ArtifactInput(
                "frame_context",
                rent.RENT_OUTPUT_CONTEXT,
                "frame_context",
                graph_context.US_FRAME_CONTEXT_TYPE,
            ),
            projection_edge,
            validation_edge,
        ),
        artifact_outputs=(
            ArtifactOutput("source_outputs", codec.HOUSING_SOURCE_OUTPUTS_TYPE),
        ),
    )
    outputs_edge = ArtifactInput(
        "source_outputs",
        HOUSING_SOURCE_OUTPUTS,
        "source_outputs",
        codec.HOUSING_SOURCE_OUTPUTS_TYPE,
    )
    merge_boundary = Node(
        HOUSING_SOURCE_MERGE_POPULATION,
        USHousingMergeBoundaryKernel.ref,
        base=base,
        structural=StructuralDelta.FILTER,
        inputs=(Slice("person", boundary_columns),),
        params=mode_params,
        artifact_inputs=(parent_context_edge, projection_edge, outputs_edge),
        artifact_outputs=source_context,
    )
    merge_outputs = tuple(
        Owned(
            entity,
            column,
            dtype,
            rewrite=(entity, column) in inventory,
        )
        for entity, column, dtype in codec.merge_columns()
    )
    merge = Node(
        HOUSING_SOURCE_MERGE,
        USHousingMergeKernel.ref,
        population=HOUSING_SOURCE_MERGE_POPULATION,
        inputs=(Slice("person", boundary_columns),),
        outputs=merge_outputs,
        params=mode_params,
        artifact_inputs=(
            ArtifactInput(
                "frame_context",
                HOUSING_SOURCE_MERGE_POPULATION,
                "frame_context",
                graph_context.US_FRAME_CONTEXT_TYPE,
            ),
            projection_edge,
            outputs_edge,
        ),
        artifact_outputs=(
            ArtifactOutput("frame_context", codec.HOUSING_PENDING_CONTEXT_TYPE),
            ArtifactOutput("merge_evidence", codec.HOUSING_MERGE_EVIDENCE_TYPE),
        ),
    )
    gate_slices = []
    for entity in codec.SOURCE_ENTITIES:
        names = [column for _e, column, _d in codec.merge_columns() if _e == entity]
        if entity == "person":
            names = [*boundary_columns, *names]
        gate_slices.append(Slice(entity, tuple(dict.fromkeys(names))))
    gate = Node(
        HOUSING_SOURCE_ACCEPTED_CONTEXT,
        USHousingSourceGateKernel.ref,
        population=HOUSING_SOURCE_MERGE_POPULATION,
        inputs=tuple(gate_slices),
        params=mode_params,
        artifact_inputs=(
            ArtifactInput(
                "parent_frame_context",
                context_producer,
                "frame_context",
                graph_context.US_FRAME_CONTEXT_TYPE,
            ),
            ArtifactInput(
                "source_frame_context",
                rent.RENT_OUTPUT_CONTEXT,
                "frame_context",
                graph_context.US_FRAME_CONTEXT_TYPE,
            ),
            ArtifactInput(
                "frame_context",
                HOUSING_SOURCE_MERGE,
                "frame_context",
                codec.HOUSING_PENDING_CONTEXT_TYPE,
            ),
            ArtifactInput(
                "merge_evidence",
                HOUSING_SOURCE_MERGE,
                "merge_evidence",
                codec.HOUSING_MERGE_EVIDENCE_TYPE,
            ),
            projection_edge,
            outputs_edge,
            validation_edge,
        ),
        artifact_outputs=source_context,
    )
    return (
        source,
        measured,
        *rent_nodes,
        source_outputs,
        merge_boundary,
        merge,
        gate,
    )


def register_us_housing_source_kernels(
    registry: KernelRegistry, *, source_codecs=None
) -> None:
    """Register this recipe's kernels together with the existing rent kernels."""
    rent.register_us_acs_rent_kernels(registry, source_codecs=source_codecs)
    for kernel in (
        USHousingSourceBoundaryKernel(),
        USHousingMeasuredKernel(),
        USHousingSourceOutputsKernel(),
        USHousingMergeBoundaryKernel(),
        USHousingMergeKernel(),
        USHousingSourceGateKernel(),
    ):
        registry.register(kernel)


def assert_merge_labels_supported(
    target: pd.DataFrame, columns: Sequence[str], *, entity: str
) -> None:
    """Compatibility spelling for the registered label-seam refusal."""
    assert_multispine_source_merge_labels_supported(
        target, columns, entity=entity, operator_name="with_us_housing_inputs"
    )
