"""Corrected ASEC monetary and demographic measurements over composed ASEC rows.

Two pure nodes hang off :mod:`.graph_composed_asec_binding`. Each writes only
under the binding's declared boolean arm mask, so the executor holds every
native ACS cell — value and missingness alike — byte-identical: a masked
``Owned`` is checked by ``population._patch_columns``, which refuses any change
to non-owned storage. Where a column is new the opposite arm keeps a declared
null; nothing is inferred as zero.

Neither kernel reads a source-channel or spine-source-ID column. They receive
the arm's rows through the declared mask and the arm's source coordinates
through the typed ``arm_rows`` artifact, both already resolved and digested by
the binding node. The arithmetic is the accepted one, unchanged and imported:
:func:`~.cps_carried_current.derive_cps_carried_current_leaves` for the
corrected leaves (which is also where ``age`` comes from, mapped from ``A_AGE``)
and :func:`~.graph_asec_income.derive_reported_income` for the six reported
observations. Raw source observations and their quality/zero/allocation axes
stay inside those artifacts; only the computed quantities become cells.

Nothing here is a release, a calibration, a transfer or a score.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json

from .asec_current_money import _sha
from .asec_current_money_selection import (
    US_ASEC_PREPARED_RECEIPT_TYPE,
    US_ASEC_SELECTED_MONEY_TYPE,
    decode_selected_current_money,
)
from .cps_carried_current import (
    CPS_CARRIED_CURRENT_PERSON_LEAVES,
    CPS_CARRIED_CURRENT_ROUTING_COLUMNS,
    CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
    cps_carried_current_leaf_contract,
    derive_cps_carried_current_leaves,
)
from .graph_asec_income import (
    RESULT_COLUMNS,
    US_ASEC_INCOME_OBSERVATIONS_TYPE,
    US_ASEC_REPORTED_INCOME_TYPE,
    bind_income_observations,
    derive_reported_income,
    encode_reported_income,
)
from .graph_asec_prepared import PreparedGraphError, _artifact, _same_producer
from .graph_composed_asec_binding import (
    BIND_NODE,
    COHORT_COLUMN,
    COMPOSED_ASEC_DEPENDENCIES,
    COMPOSED_ASEC_PHASE,
    COMPOSED_ASEC_STAGE,
    US_COMPOSED_ASEC_ARM_ROWS_TYPE,
    US_COMPOSED_ASEC_BINDING_TYPE,
    USComposedAsecBindKernel,
    arm_row_column,
    bind_composed_asec_arm_rows,
    bind_composed_asec_document,
    composed_asec_bind_node,
    refuse_unbound_demographic_writes,
)
from .graph_composed_contracts import (
    LEAVES_NODE as LEAVES_NODE,
)
from .graph_composed_contracts import (
    REPORTED_INCOME_NODE as REPORTED_INCOME_NODE,
)
from .graph_composed_population import (
    COMPOSED_SOURCES,
    CREATE_NODE,
    GEOGRAPHY_PHASE,
    HARMONIZE_NODE,
    composed_population_nodes,
    composed_population_registry,
)
from .graph_context import US_FRAME_CONTEXT_TYPE, _decode, _mass_records, _row_identity
from .graph_geography import LOOKUP_SOURCES, us_geography_nodes
from .graph_implementation import implementation_hash, implementation_manifest

LEAF_DTYPE = "float64"

#: Corrected leaves the native ACS arm already carries under the same name.
#: On those the masked write fills the ASEC rows and the executor holds every
#: ACS cell byte-identical; on every other leaf the column is new and the ACS
#: rows stay declared-null. Any other incumbent is unreviewed and refuses.
ACS_SHARED_LEAVES = (
    "age",
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparedGraphError(reason)


def _leaf_coordinates() -> tuple[tuple[str, str], ...]:
    return (
        *(("person", name) for name in CPS_CARRIED_CURRENT_PERSON_LEAVES),
        *(("spm_unit", name) for name in CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES),
    )


def composed_leaf_declarations() -> tuple[Owned, ...]:
    """The corrected-leaf cells, owned only at the bound arm's rows.

    The binding document records ``is_female`` and ``is_household_head`` as
    unbound on this population. That record stays honest only while no node of
    the stage writes them, so the declaration itself refuses a leaf roster that
    would (``ASEC_DEMOGRAPHIC_COLUMN_BOUND``) rather than letting a reviewed
    mapping quietly turn a recorded gap into an unreviewed binding.
    """
    coordinates = _leaf_coordinates()
    refuse_unbound_demographic_writes(column for _entity, column in coordinates)
    return tuple(
        Owned(entity, column, LEAF_DTYPE, rows=arm_row_column(entity))
        for entity, column in coordinates
    )


def composed_reported_income_declarations() -> tuple[Owned, ...]:
    """The six reported-income cells, owned only at the bound arm's rows."""
    refuse_unbound_demographic_writes(RESULT_COLUMNS)
    return tuple(
        Owned("person", name, LEAF_DTYPE, rows=arm_row_column("person"))
        for name in RESULT_COLUMNS
    )


def _document(value) -> dict:
    """Decode the producing node's context, keeping every graph-context check."""
    document = _decode(value.payload)
    _mass_records(document["mass_log"])
    _require(canonical_json(document) == value.payload, "CONTEXT_CANONICAL")
    return document


def _bound_population(document: dict, binding: dict) -> None:
    """The context and the binding must describe one composed population."""
    for entity in US_SCHEMA.entities:
        declared = document["entities"][entity]
        recorded = binding["composed"][entity]
        _require(
            declared["rows"] == recorded["rows"]
            and declared["ordered_ids_sha256"] == recorded["ordered_ids_sha256"],
            f"BINDING_POPULATION_IDENTITY:{entity}",
        )
    for entity, column in binding["arm_row_columns"].items():
        _require(
            column == arm_row_column(entity)
            and column in document["entities"][entity]["columns"],
            f"BINDING_ARM_COLUMN:{entity}",
        )


def _arm_rows(table: pd.DataFrame, entity: str, binding: dict) -> np.ndarray:
    """The masked view's own ids, checked against the binding's recorded arm."""
    id_column = US_SCHEMA.entity_id_column(entity)
    ids = table[id_column].to_numpy(dtype="int64")
    recorded = binding["arm"][entity]
    _require(len(ids) == recorded["rows"], f"ARM_VIEW_ROWS:{entity}")
    _require(
        _row_identity(pd.DataFrame({id_column: ids}), entity)["ordered_ids_sha256"]
        == recorded["composed_ordered_ids_sha256"],
        f"ARM_VIEW_IDENTITY:{entity}",
    )
    column = arm_row_column(entity)
    _require(
        column in table and bool(table[column].to_numpy().all()),
        f"ARM_VIEW_MASK:{entity}",
    )
    return ids


def _selected_money(binding: dict, prepared: dict, values):
    """Decode the arm's money slice against the binding it was cut for."""
    selected = decode_selected_current_money(
        values["selected_current_money"].payload,
        expected_parent_header_sha256=prepared["money_header_sha256"],
        expected_parent_content_sha256=prepared["money_content_sha256"],
        expected_prepared_receipt_sha256=_sha(values["prepared_receipt"].payload),
        expected_selection_sha256=_sha(values["asec_binding"].payload),
    )
    header = selected.header_data
    _require(
        header["person_identity_sha256"]
        == binding["arm"]["person"]["original_ordered_ids_sha256"]
        and header["household_identity_sha256"]
        == binding["arm"]["household"]["original_ordered_ids_sha256"],
        "SELECTED_ARM_COORDINATES",
    )
    _require(
        selected.person_rows == binding["arm"]["person"]["rows"]
        and selected.household_rows == binding["arm"]["household"]["rows"],
        "SELECTED_ARM_ROWS",
    )
    return selected


def _binding(values) -> dict:
    binding = bind_composed_asec_document(values["asec_binding"].payload)
    _require(
        binding["inputs"]["prepared_receipt_sha256"]
        == _sha(values["prepared_receipt"].payload),
        "BINDING_PREPARED_RECEIPT",
    )
    _require(
        binding["inputs"]["producers"]["prepared_source"]
        == values["prepared_receipt"].producer_key,
        "BINDING_SOURCE_PRODUCER",
    )
    return binding


def _prepared(value) -> dict:
    """The producing CREATE's own receipt, re-read as canonical bytes."""
    document = json.loads(value.payload)
    _require(canonical_json(document) == value.payload, "PREPARED_RECEIPT_JSON")
    _require(document["release_eligible"] is False, "PREPARED_RECEIPT_SCHEMA")
    return document


def _new_columns(document: dict, coordinates) -> None:
    """Record only the coordinates this stage actually adds to the population."""
    for entity, column in coordinates:
        columns = document["entities"][entity]["columns"]
        if column not in columns:
            columns.append(column)


class USComposedAsecLeavesKernel(KernelBase):
    """Derive the corrected monetary and age leaves over the bound arm's rows."""

    ref = "us.composed_population.asec_cps_carried_current@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=COMPOSED_ASEC_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return implementation_hash(COMPOSED_ASEC_STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(set(context.params) == {"phase"}, "NODE_PARAMS")
        _require(context.params["phase"] == COMPOSED_ASEC_PHASE, "NODE_PHASE")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        _require(not node.sources, "NODE_SOURCES")
        _require(node.inputs == _leaves_inputs(), "NODE_SLICES")
        _require(node.outputs == composed_leaf_declarations(), "NODE_OUTPUTS")
        _require(
            set(context.artifacts)
            == {
                "frame_context",
                "asec_binding",
                "arm_rows",
                "selected_current_money",
                "prepared_receipt",
            },
            "NODE_ARTIFACT_INPUTS",
        )
        _require(
            node.artifact_outputs
            == (ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
            "NODE_ARTIFACT_OUTPUTS",
        )
        values = {
            name: _artifact(context, name, kind)
            for name, kind in (
                ("frame_context", US_FRAME_CONTEXT_TYPE),
                ("asec_binding", US_COMPOSED_ASEC_BINDING_TYPE),
                ("arm_rows", US_COMPOSED_ASEC_ARM_ROWS_TYPE),
                ("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
                ("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
            )
        }
        _same_producer(
            context,
            ("frame_context", "asec_binding", "arm_rows", "selected_current_money"),
        )
        binding = _binding(values)
        prepared = _prepared(values["prepared_receipt"])
        document = _document(values["frame_context"])
        _bound_population(document, binding)
        incumbent = sorted(
            f"{entity}.{column}"
            for entity, column in _leaf_coordinates()
            if column in document["entities"][entity]["columns"]
        )
        _require(
            set(incumbent) <= {f"person.{name}" for name in ACS_SHARED_LEAVES},
            f"UNREVIEWED_LEAF_INCUMBENT:{incumbent}",
        )
        person, spm_unit = context.tables["person"], context.tables["spm_unit"]
        person_ids = _arm_rows(person, "person", binding)
        spm_ids = _arm_rows(spm_unit, "spm_unit", binding)
        selected = _selected_money(binding, prepared, values)
        rows = bind_composed_asec_arm_rows(
            values["arm_rows"].payload, binding_document=binding
        )
        _require(
            len(rows.array("person", "original_ids")) == len(person_ids),
            "ARM_ROWS_PERSON_ALIGNMENT",
        )
        leaves = derive_cps_carried_current_leaves(
            selected,
            routing=person.loc[:, list(CPS_CARRIED_CURRENT_ROUTING_COLUMNS)],
            spm_membership=person[US_SCHEMA.membership_column("spm_unit")].to_numpy(
                dtype="int64"
            ),
            spm_ids=spm_ids,
        )
        person_index = pd.Index(person_ids, name=US_SCHEMA.entity_id_column("person"))
        spm_index = pd.Index(spm_ids, name=US_SCHEMA.entity_id_column("spm_unit"))
        columns = {
            ("person", name): pd.Series(values_, index=person_index, dtype=LEAF_DTYPE)
            for name, values_ in leaves.person.items()
        }
        columns.update(
            {
                ("spm_unit", name): pd.Series(
                    values_, index=spm_index, dtype=LEAF_DTYPE
                )
                for name, values_ in leaves.spm_unit.items()
            }
        )
        _new_columns(document, _leaf_coordinates())
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": canonical_json(document)},
            receipt={
                "phase": COMPOSED_ASEC_PHASE,
                "implementation": implementation_manifest(COMPOSED_ASEC_STAGE),
                "contract": cps_carried_current_leaf_contract(),
                "binding_sha256": _sha(values["asec_binding"].payload),
                "selected_money_sha256": _sha(values["selected_current_money"].payload),
                "arm_person_rows": int(len(person_ids)),
                "arm_spm_unit_rows": int(len(spm_ids)),
                "acs_shared_incumbents": incumbent,
                "demographics": binding["demographics"],
                "release_eligible": False,
                "certified": False,
            },
        )


class USComposedAsecReportedIncomeKernel(KernelBase):
    """Reconstruct the six reported observations over the bound arm's rows."""

    ref = "us.composed_population.asec_reported_income@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=COMPOSED_ASEC_DEPENDENCIES,
    )

    def implementation_hash(self) -> str:
        return implementation_hash(COMPOSED_ASEC_STAGE)

    def run(self, context: KernelContext) -> KernelResult:
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(set(context.params) == {"phase"}, "NODE_PARAMS")
        _require(context.params["phase"] == COMPOSED_ASEC_PHASE, "NODE_PHASE")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        _require(not node.sources, "NODE_SOURCES")
        _require(node.inputs == _reported_income_inputs(), "NODE_SLICES")
        _require(
            node.outputs == composed_reported_income_declarations(), "NODE_OUTPUTS"
        )
        _require(
            set(context.artifacts)
            == {
                "frame_context",
                "asec_binding",
                "arm_rows",
                "selected_current_money",
                "prepared_receipt",
                "income_observations",
            },
            "NODE_ARTIFACT_INPUTS",
        )
        _require(
            node.artifact_outputs
            == (
                ArtifactOutput("reported_income", US_ASEC_REPORTED_INCOME_TYPE),
                ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ),
            "NODE_ARTIFACT_OUTPUTS",
        )
        values = {
            name: _artifact(context, name, kind)
            for name, kind in (
                ("frame_context", US_FRAME_CONTEXT_TYPE),
                ("asec_binding", US_COMPOSED_ASEC_BINDING_TYPE),
                ("arm_rows", US_COMPOSED_ASEC_ARM_ROWS_TYPE),
                ("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
                ("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
                ("income_observations", US_ASEC_INCOME_OBSERVATIONS_TYPE),
            )
        }
        binding_producer = _same_producer(
            context, ("asec_binding", "arm_rows", "selected_current_money")
        )
        source_producer = _same_producer(
            context, ("prepared_receipt", "income_observations")
        )
        binding = _binding(values)
        prepared = _prepared(values["prepared_receipt"])
        document = _document(values["frame_context"])
        _bound_population(document, binding)
        _require(
            not set(RESULT_COLUMNS) & set(document["entities"]["person"]["columns"]),
            "OWNED_LEAF_INCUMBENT",
        )
        person = context.tables["person"]
        person_ids = _arm_rows(person, "person", binding)
        selected = _selected_money(binding, prepared, values)
        rows = bind_composed_asec_arm_rows(
            values["arm_rows"].payload, binding_document=binding
        )
        income = bind_income_observations(
            values["income_observations"].payload, prepared_receipt=prepared
        )
        _require(
            person[COHORT_COLUMN].dtype == np.dtype("int64"), "ARM_SOURCE_YEAR_DTYPE"
        )
        result = derive_reported_income(
            selected,
            income,
            # The arm's own original person identity, resolved and digested by
            # the binding node; the composed row ids never enter the accounting.
            person_ids=rows.array("person", "original_ids"),
            income_years=person[COHORT_COLUMN].to_numpy(dtype="int64"),
            person_positions_sha256=binding["arm"]["person"]["source_positions_sha256"],
            prepared_receipt_sha256=_sha(values["prepared_receipt"].payload),
            selection_sha256=_sha(values["asec_binding"].payload),
            selected_money_sha256=_sha(values["selected_current_money"].payload),
            income_payload_sha256=_sha(values["income_observations"].payload),
            source_producer_key=source_producer,
            selection_producer_key=binding_producer,
            frame_context_sha256=_sha(values["frame_context"].payload),
        )
        index = pd.Index(person_ids, name=US_SCHEMA.entity_id_column("person"))
        columns = {
            ("person", name): pd.Series(
                result.array(name), index=index, dtype=LEAF_DTYPE
            )
            for name in RESULT_COLUMNS
        }
        _new_columns(document, tuple(("person", name) for name in RESULT_COLUMNS))
        payload = encode_reported_income(result)
        return KernelResult(
            columns=columns,
            artifacts={
                "reported_income": payload,
                "frame_context": canonical_json(document),
            },
            receipt={
                "phase": COMPOSED_ASEC_PHASE,
                "implementation": implementation_manifest(COMPOSED_ASEC_STAGE),
                "accounting_sha256": _sha(payload),
                "binding_sha256": _sha(values["asec_binding"].payload),
                "arm_person_rows": int(len(person_ids)),
                "release_eligible": False,
                "certified": False,
            },
        )


def _leaves_inputs() -> tuple[Slice, ...]:
    return (
        Slice(
            "person",
            (*CPS_CARRIED_CURRENT_ROUTING_COLUMNS, arm_row_column("person")),
            rows=arm_row_column("person"),
        ),
        Slice(
            "spm_unit",
            (arm_row_column("spm_unit"),),
            rows=arm_row_column("spm_unit"),
        ),
    )


def _reported_income_inputs() -> tuple[Slice, ...]:
    return (
        Slice(
            "person",
            (COHORT_COLUMN, arm_row_column("person")),
            rows=arm_row_column("person"),
        ),
    )


def composed_asec_measure_nodes(*, population: str, population_context: str):
    """Bind the arm, then the corrected leaves, then the reported observations."""
    bind = composed_asec_bind_node(
        population=population, population_context=population_context
    )
    binding_artifacts = (
        ArtifactInput(
            "asec_binding", BIND_NODE, "asec_binding", US_COMPOSED_ASEC_BINDING_TYPE
        ),
        ArtifactInput(
            "arm_rows", BIND_NODE, "arm_rows", US_COMPOSED_ASEC_ARM_ROWS_TYPE
        ),
        ArtifactInput(
            "selected_current_money",
            BIND_NODE,
            "selected_current_money",
            US_ASEC_SELECTED_MONEY_TYPE,
        ),
    )
    prepared_receipt = ArtifactInput(
        "prepared_receipt",
        CREATE_NODE,
        "prepared_receipt",
        US_ASEC_PREPARED_RECEIPT_TYPE,
    )
    leaves = Node(
        id=LEAVES_NODE,
        kernel=USComposedAsecLeavesKernel.ref,
        population=population,
        inputs=_leaves_inputs(),
        outputs=composed_leaf_declarations(),
        params={"phase": COMPOSED_ASEC_PHASE},
        artifact_inputs=(
            ArtifactInput(
                "frame_context", BIND_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
            ),
            *binding_artifacts,
            prepared_receipt,
        ),
        artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
    )
    reported = Node(
        id=REPORTED_INCOME_NODE,
        kernel=USComposedAsecReportedIncomeKernel.ref,
        population=population,
        inputs=_reported_income_inputs(),
        outputs=composed_reported_income_declarations(),
        params={"phase": COMPOSED_ASEC_PHASE},
        artifact_inputs=(
            ArtifactInput(
                "frame_context", LEAVES_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
            ),
            *binding_artifacts,
            prepared_receipt,
            ArtifactInput(
                "income_observations",
                CREATE_NODE,
                "income_observations",
                US_ASEC_INCOME_OBSERVATIONS_TYPE,
            ),
        ),
        artifact_outputs=(
            ArtifactOutput("reported_income", US_ASEC_REPORTED_INCOME_TYPE),
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
        ),
    )
    return (bind, leaves, reported)


def composed_asec_population_graph(
    columns,
    *,
    sample_fraction: float,
    sample_seed: int,
    geography: bool = True,
) -> Graph:
    """The composed development population plus this arm's bound measurements."""
    composed = composed_population_nodes(
        columns, sample_fraction=sample_fraction, sample_seed=sample_seed
    )
    if not geography:
        return Graph(
            country="us",
            sources=COMPOSED_SOURCES,
            nodes=(
                *composed,
                *composed_asec_measure_nodes(
                    population=HARMONIZE_NODE, population_context=HARMONIZE_NODE
                ),
            ),
        )
    geography_nodes = us_geography_nodes(
        columns, base=HARMONIZE_NODE, context_producer=HARMONIZE_NODE
    )
    return Graph(
        country="us",
        sources=(*COMPOSED_SOURCES, *LOOKUP_SOURCES),
        nodes=(
            *composed,
            *geography_nodes,
            *composed_asec_measure_nodes(
                population=f"{GEOGRAPHY_PHASE}.boundary",
                population_context=GEOGRAPHY_PHASE,
            ),
        ),
    )


def composed_asec_population_registry(*, geography: bool = True) -> KernelRegistry:
    """The composed registry plus exactly this stage's three kernels."""
    registry = composed_population_registry(geography=geography)
    registry.register(USComposedAsecBindKernel())
    registry.register(USComposedAsecLeavesKernel())
    registry.register(USComposedAsecReportedIncomeKernel())
    return registry


__all__ = [
    "ACS_SHARED_LEAVES",
    "LEAF_DTYPE",
    "LEAVES_NODE",
    "REPORTED_INCOME_NODE",
    "USComposedAsecLeavesKernel",
    "USComposedAsecReportedIncomeKernel",
    "composed_asec_measure_nodes",
    "composed_asec_population_graph",
    "composed_asec_population_registry",
    "composed_leaf_declarations",
    "composed_reported_income_declarations",
]
