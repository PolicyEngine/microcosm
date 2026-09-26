"""Graph nodes that run the pure PUF growth transform inside a real population.

Three nodes, in the shape the US build already uses for a source operation:

``<stage>.boundary``
    A ``FILTER`` node that keeps every person. It exists because a column a
    ``CREATE`` node loaded can only be rewritten once a structural node has
    opened a new version. Each kernel reconstructs the declared contract from
    its parameters before its computation. The boundary's receipt states the
    contract digest, the factor authority, and whether that authority could
    ever stand behind a release.

``<stage>.money``
    Owns each declared money column as a rewrite, plus the two provenance
    columns. Its whole computation is
    :func:`~microcosm.build.us_runtime.puf_growth.apply_puf_growth`.

``<stage>.design_weight``
    Owns only the declared design-weight column, through
    :func:`~microcosm.build.us_runtime.puf_growth.apply_puf_design_weight_growth`.
    A separate node so the graph itself records that money growth and weight
    growth are different operations reading different factor series.

The contract travels as node parameters — the recipe identity document and the
factor-table document, verbatim — so a changed contract changes the node key
and nothing is served from the store across a contract change. Codes,
identifiers and counts are read but never owned. Two executor mechanisms are
what actually keep them byte-identical: a kernel's returned coordinates must be
exactly the node's declared outputs, so it cannot write them at all; and the
projected input view is frozen and its digest compared before and after the
kernel runs, so it cannot mutate them in place either.

The design weight this stage grows is the **raw source column**. Promoting it
into a frame's typed design weights is a separate, later obligation with its
own mass policy; nothing here touches ``Frame`` weights. The weight column is
owned as ``float64``, as are the money rewrites; a base version carrying any
of those columns as an integer would need a
declared dtype change, which is a reviewed decision rather than a cast here.

**The closed read set is the transform's, not the frame's.**
:func:`~microcosm.build.us_runtime.puf_growth.apply_puf_growth` refuses a table
carrying a column no rule declares. These nodes cannot make that promise about
the *population*: a ``CREATE`` node upstream may have loaded raw money columns
this contract never mentions, and they pass through the stage ungrown, because a
node only sees the slices it declares. Closing that gap needs a source-stage
contract that declares the full raw column inventory and checks the population
against it; that is a documented remaining obligation, not something this stage
silently covers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import import_module

import pandas as pd

from microcosm.build.us_runtime.puf_growth import (
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
    CompiledPufGrowth,
    GrowthFactorTable,
    PufGrowthRefusalError,
    apply_puf_design_weight_growth,
    apply_puf_growth,
    compile_puf_growth,
    parse_growth_recipe,
    require_transformable,
)
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
    source_hash,
)

__all__ = [
    "PUF_GROWTH_BOUNDARY_NODE",
    "PUF_GROWTH_DESIGN_WEIGHT_NODE",
    "PUF_GROWTH_MONEY_NODE",
    "PUF_GROWTH_PROVENANCE_RECID_DTYPE",
    "PUF_GROWTH_STAGE",
    "PufGrowthBoundaryKernel",
    "PufGrowthDesignWeightKernel",
    "PufGrowthMoneyKernel",
    "register_us_puf_growth_kernels",
    "us_puf_growth_nodes",
]

#: Stage name, and the node ids derived from it.
PUF_GROWTH_STAGE = "us_puf_growth"
PUF_GROWTH_BOUNDARY_NODE = f"{PUF_GROWTH_STAGE}.boundary"
PUF_GROWTH_MONEY_NODE = f"{PUF_GROWTH_STAGE}.money"
PUF_GROWTH_DESIGN_WEIGHT_NODE = f"{PUF_GROWTH_STAGE}.design_weight"

#: The dtype the provenance record identifier is owned as. The raw identifier
#: is an integer record id; a float copy of it would not be a faithful key.
PUF_GROWTH_PROVENANCE_RECID_DTYPE = "int64"

#: Modules whose bytes are attested by every kernel's implementation hash. The
#: pure contract is included because a change there changes what these kernels
#: compute even when this file is untouched.
_IMPLEMENTATION_MODULES = (
    "microcosm.build.monetary_targets",
    "microcosm.build.us_runtime.puf_growth",
    "microcosm.build.us_runtime.puf_growth_graph",
    "microcosm.build.us_runtime.puf_source_agi",
)

_PARAMS = ("entity", "factor_table", "recipe")


class _PufGrowthKernel(KernelBase):
    """Shared parameter contract and code identity for the three kernels."""

    def implementation_hash(self) -> str:
        # Keep the declared dependencies in the hash the way KernelBase does:
        # without them a declared distribution would bind only its name into
        # the node key, never its installed version.
        return source_hash(
            *(import_module(name) for name in _IMPLEMENTATION_MODULES),
            dependencies=self.capabilities.dependencies,
        )

    @staticmethod
    def contract(params: Mapping[str, object]) -> tuple[str, CompiledPufGrowth]:
        """Recompile the declared contract from the node's own parameters."""
        if set(params) != set(_PARAMS):
            raise PufGrowthRefusalError("NODE_PARAMETER_CONTRACT")
        entity = params["entity"]
        recipe_document = params["recipe"]
        factor_document = params["factor_table"]
        if not (
            type(entity) is str
            and type(recipe_document) is str
            and type(factor_document) is str
        ):
            raise PufGrowthRefusalError("NODE_PARAMETER_CONTRACT")
        recipe = parse_growth_recipe(recipe_document.encode("ascii"))
        factors = GrowthFactorTable.from_bytes(factor_document.encode("ascii"))
        return entity, compile_puf_growth(recipe, factors)


def _identifier_index(context: KernelContext, entity: str) -> pd.Index:
    column = f"{entity}_id"
    table = context.tables[entity]
    if column not in table:
        raise PufGrowthRefusalError("ENTITY_ID_COLUMN", entity)
    return pd.Index(table[column], name=column)


def _declared_source(context: KernelContext, entity: str, fields: Sequence[str]):
    table = context.tables[entity]
    missing = [name for name in fields if name not in table.columns]
    if missing:
        raise PufGrowthRefusalError("MISSING_DECLARED_FIELD", missing[0])
    return table.loc[:, list(fields)]


def _contract_receipt(compiled: CompiledPufGrowth) -> dict[str, object]:
    return {
        "contract_sha256": compiled.sha256,
        "factor_table_sha256": compiled.factors.sha256,
        "factor_table_id": compiled.factors.table_id,
        "factor_authority": str(compiled.factors.authority),
        "release_eligible": compiled.release_eligible,
        "target_year": compiled.target_year,
    }


class PufGrowthBoundaryKernel(_PufGrowthKernel):
    """Open a version for the rewrite, and publish the contract it will use."""

    ref = "us.puf.growth.boundary@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        _, compiled = self.contract(context.params)
        person = context.tables["person"]
        return KernelResult(
            keep=pd.Series(
                True, index=pd.Index(person.person_id, name="person_id"), dtype=bool
            ),
            receipt={"puf_growth_contract": _contract_receipt(compiled)},
        )


class PufGrowthMoneyKernel(_PufGrowthKernel):
    """Grow every declared money field once, and copy the provenance columns."""

    ref = "us.puf.growth.money@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        entity, compiled = self.contract(context.params)
        source = _declared_source(context, entity, compiled.money_view_fields)
        result = apply_puf_growth(source, compiled)
        ids = _identifier_index(context, entity)
        grown = result.table
        if grown[PROVENANCE_RECID_COLUMN].dtype != PUF_GROWTH_PROVENANCE_RECID_DTYPE:
            raise PufGrowthRefusalError(
                "PROVENANCE_RECID_DTYPE", PROVENANCE_RECID_COLUMN
            )
        columns = {
            (entity, name): pd.Series(
                grown[name].to_numpy(copy=True), index=ids, dtype="float64"
            )
            for name in compiled.money_fields
        }
        columns[(entity, PROVENANCE_RECID_COLUMN)] = pd.Series(
            grown[PROVENANCE_RECID_COLUMN].to_numpy(copy=True),
            index=ids,
            dtype=PUF_GROWTH_PROVENANCE_RECID_DTYPE,
        )
        columns[(entity, PROVENANCE_SOURCE_AGI_COLUMN)] = pd.Series(
            grown[PROVENANCE_SOURCE_AGI_COLUMN].to_numpy(copy=True),
            index=ids,
            dtype="float64",
        )
        return KernelResult(
            columns=columns,
            receipt={"puf_growth": json.loads(result.receipt_bytes)},
        )


class PufGrowthDesignWeightKernel(_PufGrowthKernel):
    """Grow the raw design-weight column, and nothing else."""

    ref = "us.puf.growth.design_weight@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context: KernelContext) -> KernelResult:
        entity, compiled = self.contract(context.params)
        rule = compiled.design_weight_rule
        if rule is None:
            raise PufGrowthRefusalError("NO_DESIGN_WEIGHT_RULE")
        source = _declared_source(context, entity, (rule.field,))
        grown = apply_puf_design_weight_growth(
            source[rule.field].to_numpy(dtype="float64"), compiled
        )
        factor = compiled.factor_for(rule.field, rule.sign_branch)
        return KernelResult(
            columns={
                (entity, rule.field): pd.Series(
                    grown, index=_identifier_index(context, entity), dtype="float64"
                )
            },
            receipt={
                "puf_design_weight_growth": {
                    **_contract_receipt(compiled),
                    "field": rule.field,
                    "series": factor.series,
                    "source_year": factor.source_year,
                    "ratio_bits": factor.ratio_bits,
                    "rows": int(len(grown)),
                }
            },
        )


def us_puf_growth_nodes(
    compiled: CompiledPufGrowth,
    *,
    base: str,
    entity: str,
    person_boundary_columns: Sequence[str],
    stage: str = PUF_GROWTH_STAGE,
) -> tuple[Node, ...]:
    """Declare the boundary, money and design-weight nodes for one contract.

    Args:
        compiled: The contract these nodes carry. Its recipe identity and its
            factor document become node parameters verbatim, so the node keys
            move whenever the contract moves.
        base: Id of the population version to filter.
        entity: Entity whose table carries the raw source fields.
        person_boundary_columns: Person columns the boundary reads. A ``FILTER``
            node returns a person mask, so it must declare a person slice.
        stage: Node-id prefix, for a graph that runs more than one contract.

    Returns:
        The boundary node, the money node, and the design-weight node.

    Raises:
        PufGrowthRefusalError: If the contract declares no design weight, names
            no person boundary column, or could not run at all
            (:func:`~microcosm.build.us_runtime.puf_growth.require_transformable`).
    """
    if type(compiled) is not CompiledPufGrowth:
        raise PufGrowthRefusalError("COMPILED_TYPE")
    weight_rule = compiled.design_weight_rule
    if weight_rule is None:
        raise PufGrowthRefusalError("NO_DESIGN_WEIGHT_RULE")
    if not person_boundary_columns:
        raise PufGrowthRefusalError("PERSON_BOUNDARY_COLUMNS")
    # Everything the money kernel needs is checked here, at declaration time,
    # rather than after a graph run has already loaded a source and written the
    # design-weight column.
    require_transformable(compiled)
    params = {
        "entity": entity,
        "factor_table": compiled.factors.document.decode("ascii"),
        "recipe": compiled.recipe.identity.decode("ascii"),
    }
    money_view = Slice(entity, compiled.money_view_fields)
    boundary = f"{stage}.boundary"
    return (
        Node(
            boundary,
            PufGrowthBoundaryKernel.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(Slice("person", tuple(person_boundary_columns)),),
            params=params,
            description="Open a version so the declared money columns can be rewritten.",
        ),
        Node(
            f"{stage}.money",
            PufGrowthMoneyKernel.ref,
            population=boundary,
            inputs=(money_view,),
            outputs=(
                *(
                    Owned(entity, name, "float64", rewrite=True)
                    for name in compiled.money_fields
                ),
                Owned(
                    entity,
                    PROVENANCE_RECID_COLUMN,
                    PUF_GROWTH_PROVENANCE_RECID_DTYPE,
                ),
                Owned(entity, PROVENANCE_SOURCE_AGI_COLUMN, "float64"),
            ),
            params=params,
            description="Grow declared money once and carry source provenance.",
        ),
        Node(
            f"{stage}.design_weight",
            PufGrowthDesignWeightKernel.ref,
            population=boundary,
            inputs=(Slice(entity, (weight_rule.field,)),),
            outputs=(Owned(entity, weight_rule.field, "float64", rewrite=True),),
            params=params,
            description="Grow the raw design weight by its own declared series.",
        ),
    )


def register_us_puf_growth_kernels(registry: KernelRegistry) -> None:
    """Register the three PUF growth kernels."""
    registry.register(PufGrowthBoundaryKernel())
    registry.register(PufGrowthMoneyKernel())
    registry.register(PufGrowthDesignWeightKernel())
