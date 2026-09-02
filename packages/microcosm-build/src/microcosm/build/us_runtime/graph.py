"""The stacked US pool's post-transfer stages as an executable graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from microcosm.graph import (
    Graph,
    KernelRegistry,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
)

from .capital_gain_distributions import (
    capital_gain_distribution_shares_asset_identity,
)
from .multispine_pool import (
    POOL_ENGINE_INPUT_PROJECTION_CONTRACT,
    POOL_RANDOM_SEED,
    POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256,
    POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
    POOL_SSI_DEPENDENCY_CONTRACT,
    POOL_TIME_PERIOD,
)
from .qbi_inputs import (
    US_QBI_BOOLEAN_OUTPUT_COLUMNS,
    US_QBI_OUTPUT_COLUMNS,
    US_QBI_RECONCILED_PERSON_COLUMNS,
    us_qbi_reconciliation_contract_identity,
)
from .take_up_contract import load_take_up_contract, take_up_contract_identity

__all__ = ["us_post_transfer_graph", "us_registry"]


_SOURCE = SourceRef(
    "stacked",
    "csv-tables",
    "Synthetic stacked US pool at the completed ACS-transfer boundary.",
)
_ENGINE_REF = "policyengine-us"


@dataclass(frozen=True)
class _Cell:
    entity: str
    column: str
    dtype: str

    @property
    def coordinate(self) -> tuple[str, str]:
        return self.entity, self.column

    def owned(self, *, rewrite: bool = False) -> Owned:
        return Owned(self.entity, self.column, self.dtype, rewrite=rewrite)


def _cells(
    entity: str,
    columns: Iterable[str],
    dtype: str,
) -> tuple[_Cell, ...]:
    return tuple(_Cell(entity, column, dtype) for column in columns)


def _take_up_cells(
    entity: str,
    *,
    include_eitc: bool,
) -> tuple[_Cell, ...]:
    return tuple(
        _Cell(program.entity, program.variable, "bool")
        for program in load_take_up_contract().programs
        if program.entity == entity
        and (include_eitc or program.variable != "takes_up_eitc")
    )


_QBI_BOOLEAN_COLUMNS = frozenset(US_QBI_BOOLEAN_OUTPUT_COLUMNS)
_QBI_ROOT_CELLS = tuple(
    _Cell(
        "person",
        column,
        "bool" if column in _QBI_BOOLEAN_COLUMNS else "float64",
    )
    for column in US_QBI_OUTPUT_COLUMNS
)

_ROOT_CELLS = (
    *_cells("person", ("A_AGE",), "float64"),
    *_cells("person", ("A_SEX", "source_year"), "int64"),
    *_cells("person", ("measured",), "float64"),
    # The unchanged stage validators inspect source and support provenance on
    # every entity, so the CREATE node loads those columns; no kernel here reads
    # them (test_us_spine_blindness lists this module as a declaration-only
    # provenance owner for that reason).
    *_cells(
        "person",
        ("person_spine_source_id", "person_source_id"),
        "int64",
    ),
    *_cells("person", ("person_support_channel",), "string"),
    *_cells("person", ("person_support_clone_index",), "int64"),
    *_cells(
        "person",
        (
            "age",
            "SEMP",
            "long_term_capital_gains_before_response",
            "non_sch_d_capital_gains",
            "schedule_d_capital_gain_distributions",
            "self_employment_income_before_lsr",
            "non_qualified_dividend_income",
        ),
        "float64",
    ),
    *_QBI_ROOT_CELLS,
    *_take_up_cells("person", include_eitc=False),
    *_cells(
        "person",
        ("short_term_capital_gains", "long_term_capital_gains_on_collectibles"),
        "float64",
    ),
    *_cells("household", ("TYPEHUGQ",), "int64"),
    *_cells(
        "household",
        ("household_spine_source_id", "household_source_id"),
        "int64",
    ),
    *_cells("household", ("household_support_channel",), "string"),
    *_cells("household", ("household_support_clone_index",), "int64"),
    *_cells(
        "tax_unit",
        ("tax_unit_spine_source_id", "tax_unit_source_id"),
        "int64",
    ),
    *_cells("tax_unit", ("tax_unit_support_channel",), "string"),
    *_cells("tax_unit", ("tax_unit_support_clone_index",), "int64"),
    *_take_up_cells("tax_unit", include_eitc=False),
    *_cells(
        "tax_unit",
        (
            "puf_capital_gains_tail_transfer_applied",
            "puf_capital_gains_tail_donor_is_synthetic",
        ),
        "bool",
    ),
    *_cells(
        "tax_unit",
        (
            "puf_capital_gains_tail_donor_source_id",
            "puf_capital_gains_tail_donor_filing_status_code",
            "puf_capital_gains_tail_donor_agi_band_index",
        ),
        "int64",
    ),
    *_cells(
        "tax_unit",
        (
            "puf_capital_gains_tail_transfer_weight",
            "unrecaptured_section_1250_gain",
        ),
        "float64",
    ),
    *_cells(
        "spm_unit",
        ("spm_unit_spine_source_id", "spm_unit_source_id"),
        "int64",
    ),
    *_cells("spm_unit", ("spm_unit_support_channel",), "string"),
    *_cells("spm_unit", ("spm_unit_support_clone_index",), "int64"),
    *_take_up_cells("spm_unit", include_eitc=False),
    *_cells(
        "family",
        ("family_spine_source_id", "family_source_id"),
        "int64",
    ),
    *_cells("family", ("family_support_channel",), "string"),
    *_cells("family", ("family_support_clone_index",), "int64"),
    *_cells(
        "marital_unit",
        ("marital_unit_spine_source_id", "marital_unit_source_id"),
        "int64",
    ),
    *_cells("marital_unit", ("marital_unit_support_channel",), "string"),
    *_cells("marital_unit", ("marital_unit_support_clone_index",), "int64"),
)

_PREPARE_OUTPUTS = (
    _Cell("person", "schedule_d_capital_gain_distributions", "float64"),
)
_DERIVE_OUTPUTS = (
    *_PREPARE_OUTPUTS,
    *(
        _Cell(
            "person",
            column,
            "bool" if column in _QBI_BOOLEAN_COLUMNS else "float64",
        )
        for column in US_QBI_RECONCILED_PERSON_COLUMNS
    ),
)
_SEED_OUTPUTS = tuple(
    _Cell(program.entity, program.variable, "bool")
    for program in load_take_up_contract().programs
)
_MATERIALIZE_OUTPUTS = (_Cell("person", "ssi", "float64"),)


def _slices(
    live: Mapping[tuple[str, str], _Cell],
    *,
    exclude: Iterable[tuple[str, str]] = (),
) -> tuple[Slice, ...]:
    excluded = frozenset(exclude)
    by_entity: dict[str, list[str]] = {}
    for coordinate, cell in live.items():
        if coordinate in excluded:
            continue
        by_entity.setdefault(cell.entity, []).append(cell.column)
    return tuple(Slice(entity, tuple(columns)) for entity, columns in by_entity.items())


def _boundary(
    stage: str,
    *,
    base: str,
    live: Mapping[tuple[str, str], _Cell],
) -> Node:
    return Node(
        id=f"{stage}.boundary",
        kernel="us.post_transfer.identity@1",
        inputs=_slices(live),
        structural=StructuralDelta.FILTER,
        base=base,
        description=f"Open a population version for {stage} rewrites.",
    )


def us_post_transfer_graph() -> Graph:
    """Declare the four unchanged stages after stacked ACS transfer."""

    schedule_identity = capital_gain_distribution_shares_asset_identity()
    qbi_identity = us_qbi_reconciliation_contract_identity()
    take_up_contract = load_take_up_contract()
    take_up_identity = take_up_contract_identity(take_up_contract)

    live = {cell.coordinate: cell for cell in _ROOT_CELLS}
    nodes: list[Node] = [
        Node(
            id="create_stacked_pool",
            kernel="us.post_transfer.create@1",
            outputs=tuple(cell.owned() for cell in _ROOT_CELLS),
            structural=StructuralDelta.CREATE,
            sources=("stacked",),
            params={"context_schema_version": 1},
            description="Load the synthetic stacked pool after ACS transfer.",
        )
    ]

    prepare_boundary = _boundary(
        "prepare_stacked_tail_derivation",
        base="create_stacked_pool",
        live=live,
    )
    nodes.extend(
        (
            prepare_boundary,
            Node(
                id="prepare_stacked_tail_derivation",
                kernel="us.post_transfer.prepare@1",
                inputs=_slices(
                    live,
                    exclude=(cell.coordinate for cell in _PREPARE_OUTPUTS),
                ),
                outputs=tuple(cell.owned(rewrite=True) for cell in _PREPARE_OUTPUTS),
                population=prepare_boundary.id,
                sources=("stacked",),
                params={"stage": "prepare_stacked_tail_derivation"},
                description="Clear clone-2 Schedule-D incumbents before derivation.",
            ),
        )
    )

    derive_boundary = _boundary(
        "derive_multispine_pool_inputs",
        base=prepare_boundary.id,
        live=live,
    )
    nodes.extend(
        (
            derive_boundary,
            Node(
                id="derive_multispine_pool_inputs",
                kernel="us.post_transfer.derive@1",
                inputs=_slices(
                    live,
                    exclude=(cell.coordinate for cell in _DERIVE_OUTPUTS),
                ),
                outputs=tuple(cell.owned(rewrite=True) for cell in _DERIVE_OUTPUTS),
                population=derive_boundary.id,
                sources=("stacked",),
                params={
                    "stage": "derive_multispine_pool_inputs",
                    "remaining_stage_manifest_sha256": (
                        POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256
                    ),
                    "schedule_d_asset_sha256": str(schedule_identity["asset_sha256"]),
                    "qbi_contract_version": int(qbi_identity["version"]),
                },
                description="Derive Schedule-D and reconcile the QBI input family.",
            ),
        )
    )

    seed_boundary = _boundary(
        "seed_multispine_pool_inputs",
        base=derive_boundary.id,
        live=live,
    )
    nodes.append(seed_boundary)
    seed_outputs = tuple(
        cell.owned(rewrite=cell.coordinate in live) for cell in _SEED_OUTPUTS
    )
    nodes.append(
        Node(
            id="seed_multispine_pool_inputs",
            kernel="us.post_transfer.seed@1",
            inputs=_slices(
                live,
                exclude=(cell.coordinate for cell in _SEED_OUTPUTS),
            ),
            outputs=seed_outputs,
            population=seed_boundary.id,
            sources=("stacked",),
            params={
                "stage": "seed_multispine_pool_inputs",
                "engine_ref": _ENGINE_REF,
                "seed": POOL_RANDOM_SEED,
                "time_period": POOL_TIME_PERIOD,
                "take_up_contract_version": int(take_up_identity["version"]),
                "take_up_resource_sha256": str(take_up_identity["resource_sha256"]),
            },
            description="Seed take-up inputs and disclose remaining defaults.",
        )
    )
    for cell in _SEED_OUTPUTS:
        live[cell.coordinate] = cell

    nodes.append(
        Node(
            id="materialize_multispine_agreement_outputs",
            kernel="us.post_transfer.materialize@1",
            inputs=_slices(live),
            outputs=tuple(cell.owned() for cell in _MATERIALIZE_OUTPUTS),
            population=seed_boundary.id,
            sources=("stacked",),
            params={
                "stage": "materialize_multispine_agreement_outputs",
                "engine_ref": _ENGINE_REF,
                "time_period": POOL_TIME_PERIOD,
                "household_batch_size": POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
                "ssi_dependency_sha256": POOL_SSI_DEPENDENCY_CONTRACT.sha256,
                "engine_input_projection_sha256": (
                    POOL_ENGINE_INPUT_PROJECTION_CONTRACT.sha256
                ),
                "engine_input_defaults_sha256": (
                    POOL_ENGINE_INPUT_PROJECTION_CONTRACT.defaults_sha256
                ),
            },
            description="Materialize SSI on the disposable agreement view.",
        )
    )

    return Graph(country="us", sources=(_SOURCE,), nodes=tuple(nodes))


def us_registry(*, engine: object | None = None) -> KernelRegistry:
    """Bind the post-transfer kernels to one PolicyEngine-US adapter."""

    from .graph_kernels import build_us_post_transfer_registry

    return build_us_post_transfer_registry(
        us_post_transfer_graph(),
        engine=engine,
        engine_ref=_ENGINE_REF,
    )
