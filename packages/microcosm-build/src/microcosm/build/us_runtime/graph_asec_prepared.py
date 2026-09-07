"""The five-node ASEC slice: source, selection, leaves, engine and reported income.

One directory source holds the reviewed restoration inputs. The CREATE kernel
prepares the whole authenticated population from it; a FILTER kernel draws a
seeded whole-household engineering sample and slices the money body to exactly
the selected coordinates; a pure kernel derives the corrected monetary CPS
leaves from that slice alone; and a final kernel runs the real PolicyEngine-US
adapter for the four outputs whose complete input closure this slice produces.

Two boundaries are load-bearing. The authenticated ``ReadyCurrentMoney`` never
leaves the CREATE process: what travels is its canonical encoded body, typed as
a subset artifact that no ``type(x) is ReadyCurrentMoney`` check accepts. And
engine outputs are formula-owned, so they are retained in an evaluation
artifact and never written back as population cells.

Nothing here is a release, a calibration, a representative sample, or a tax or
benefit score. Every receipt says so.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from microcosm.build.frame_sampling import EXACT_COUNT_RULE, sample_frame_households
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)
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
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
)
from microcosm.graph.canonical import canonical_json

from . import asec_housing_status_source as housing_source
from . import graph_context
from .asec_current_money import _sha
from .asec_current_money_graph_resources import load_graph_current_money_consumers
from .asec_current_money_selection import (
    US_ASEC_CURRENT_MONEY_BODY_TYPE,
    US_ASEC_PREPARED_RECEIPT_TYPE,
    US_ASEC_SELECTED_MONEY_TYPE,
    US_ASEC_SELECTION_TYPE,
    decode_selected_current_money,
    encode_selected_current_money,
    parse_current_money_body,
    select_current_money,
)
from .asec_engine_evaluation import (
    ADMITTED_ROOTS,
    BLOCKED_ENGINE_OUTPUTS,
    US_ASEC_ENGINE_EVALUATION_TYPE,
    admit_engine_outputs,
    encode_engine_evaluation,
    engine_runtime_identity,
    materialize_engine_outputs,
)
from .asec_prepared_source import (
    PREPARED_SOURCE_FILES,
    PREPARED_SOURCE_KIND,
    prepare_asec_current_money_population,
)
from .cps_carried_current import (
    CPS_CARRIED_CURRENT_PERSON_LEAVES,
    CPS_CARRIED_CURRENT_ROUTING_COLUMNS,
    CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
    cps_carried_current_leaf_contract,
    derive_cps_carried_current_leaves,
)
from .graph_asec_income import (
    REPORTED_INCOME_NODE,
    RESULT_COLUMNS,
    US_ASEC_INCOME_OBSERVATIONS_TYPE,
    US_ASEC_REPORTED_INCOME_TYPE,
    USAsecReportedIncomeKernel,
    reported_income_declarations,
)
from .graph_context import US_FRAME_CONTEXT_TYPE
from .graph_housing_universe import (
    HOUSEHOLD_EVIDENCE_COLUMNS,
    US_ASEC_HOUSING_UNIVERSE_TYPE,
    bind_housing_universe,
    verify_graph_housing_rows,
)
from .graph_implementation import (
    STAGE_DEPENDENCIES,
    implementation_hash,
    implementation_manifest,
)
from .graph_sources import frame_column_declarations

ASEC_PREPARED_STAGE = "asec_prepared_v3"
ASEC_PREPARED_PHASE = "prepare_asec_current_money_slice"
ASEC_PREPARED_CODEC = "us-asec-prepared-current-money-v3"
ASEC_PREPARED_SOURCE_NAME = "asec_prepared"
ASEC_PREPARED_SOURCE = SourceRef(
    ASEC_PREPARED_SOURCE_NAME,
    ASEC_PREPARED_CODEC,
    "Reviewed ASEC restoration inputs: P, H, T with its receipt, three housing "
    "cohort HDFs and three official Census PERSON CSV members.",
)
ASEC_PREPARED_DEPENDENCIES = STAGE_DEPENDENCIES[ASEC_PREPARED_STAGE]

PREFIX = "asec_prepared"
CREATE_NODE = f"{PREFIX}.create"
SELECT_NODE = f"{PREFIX}.select_households"
LEAVES_NODE = f"{PREFIX}.cps_carried_current"
ENGINE_NODE = f"{PREFIX}.engine_current_money"

#: The one household column the selection and engine nodes declare. They need
#: the household table and its design weights; the executor exposes an entity's
#: weights only to a node that declares that entity. This is the first column
#: the reviewed housing attachment guarantees, so a changed attachment fails
#: closed here instead of silently reading something else. It is a carrier: it
#: never reaches the corrected leaves and never reaches the engine frame.
HOUSEHOLD_WEIGHT_CARRIER = housing_source.ATTACHED_COLUMNS[0]

SELECTION_SCHEMA = "microcosm.us.asec_household_selection.v2"
SELECTION_SOURCE_LABEL = "US ASEC prepared current money"
SELECTION_STRATUM_COLUMN = "source_year"
LEAF_DTYPE = "float64"
_ENTITIES = US_SCHEMA.entities
_PERSON_ID = US_SCHEMA.entity_id_column("person")


class PreparedGraphError(ValueError):
    """A declaration, binding or identity this slice refuses to execute on."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparedGraphError(reason)


def _artifact(context: KernelContext, name: str, expected):
    """Return one declared typed artifact, refusing an aliased or retyped edge."""
    value = context.artifacts.get(name)
    edges = [edge for edge in context.node.artifact_inputs if edge.name == name]
    _require(
        value is not None
        and len(edges) == 1
        and edges[0].type == expected
        and value.type == expected,
        f"ARTIFACT_EDGE:{name}",
    )
    _require(
        isinstance(value.producer_key, str) and len(value.producer_key) == 64,
        f"ARTIFACT_PRODUCER:{name}",
    )
    return value


def _same_producer(context: KernelContext, names: Sequence[str]) -> str:
    """Require a set of edges to come from one actual producing node execution."""
    producers = {context.artifacts[name].producer_key for name in names}
    edges = {edge.name: edge.producer for edge in context.node.artifact_inputs}
    _require(len(producers) == 1, "ARTIFACT_PRODUCER_SPLIT")
    _require(len({edges[name] for name in names}) == 1, "ARTIFACT_PRODUCER_SPLIT")
    return next(iter(producers))


def _phase(context: KernelContext, extra: tuple[str, ...] = ()) -> None:
    _require(set(context.params) == {"phase", *extra}, "NODE_PARAMS")
    _require(context.params["phase"] == ASEC_PREPARED_PHASE, "NODE_PHASE")


def _document(value) -> dict:
    document = graph_context._decode(value.payload)
    graph_context._mass_records(document["mass_log"])
    _require(canonical_json(document) == value.payload, "CONTEXT_CANONICAL")
    _require(document["metadata"] == {}, "PREPARED_CONTEXT_METADATA")
    _require(document["mass_log"] == [], "PREPARED_CONTEXT_MASS_LOG")
    _require(document["weight_sources"] == {"household": "design"}, "WEIGHT_AUTHORITY")
    return document


def _identity(ids: np.ndarray, entity: str) -> dict[str, object]:
    """Row identity of an id vector, using the graph context's own digest."""
    column = US_SCHEMA.entity_id_column(entity)
    return graph_context._row_identity(
        pd.DataFrame({column: np.asarray(ids, dtype="int64")}), entity
    )


def _check_identity(document: dict, entity: str, table: pd.DataFrame) -> None:
    declared = document["entities"][entity]
    actual = graph_context._row_identity(table, entity)
    _require(
        all(declared[key] == value for key, value in actual.items()),
        f"CONTEXT_IDENTITY:{entity}",
    )
    _require(
        not set(table.columns) - set(declared["columns"]), f"CONTEXT_COLUMNS:{entity}"
    )


def _group_ids(person: pd.DataFrame, group: str) -> np.ndarray:
    """The exact id inventory a group table holds for these persons.

    ``Frame`` validates that a group table's ids are the sorted distinct values
    of the person membership column, and ``Frame.select`` prunes to exactly
    that set. Reconstructing the inventory from membership is therefore the
    same operation, not an approximation; every reconstruction below is checked
    against the producer's typed context before it is used.
    """
    membership = person[US_SCHEMA.membership_column(group)].to_numpy(dtype="int64")
    return np.unique(membership)


def _minimal_frame(person: pd.DataFrame, household: pd.DataFrame, weights, strata):
    """A person/household view for the sampler; no group beyond household."""
    columns = [_PERSON_ID, US_SCHEMA.membership_column("household")]
    extra = [name for name in (SELECTION_STRATUM_COLUMN,) if name in person]
    return Frame(
        {
            "person": person.loc[:, columns + extra].copy(deep=True),
            "household": household.loc[
                :, [US_SCHEMA.entity_id_column("household")]
            ].copy(deep=True),
        },
        EntitySchema(group_entities=("household",)),
        {
            "household": Weights(
                np.asarray(weights.values, dtype="float64"), weights.kind
            )
        },
        strata.copy(deep=True),
    )


def _household_strata(person: pd.DataFrame, household_ids: np.ndarray) -> np.ndarray:
    """One declared source-year stratum per household row; mixed years refuse."""
    years = person[SELECTION_STRATUM_COLUMN]
    _require(pd.api.types.is_integer_dtype(years.dtype), "STRATUM_DTYPE")
    _require(not bool(years.isna().any()), "STRATUM_MISSING")
    frame = pd.DataFrame(
        {
            "household": person[US_SCHEMA.membership_column("household")].to_numpy(
                dtype="int64"
            ),
            "year": years.to_numpy(dtype="int64"),
        }
    )
    distinct = frame.drop_duplicates()
    _require(not bool(distinct["household"].duplicated().any()), "MIXED_YEAR_HOUSEHOLD")
    lookup = dict(
        zip(distinct["household"].tolist(), distinct["year"].tolist(), strict=True)
    )
    _require(set(lookup) == set(household_ids.tolist()), "STRATUM_COVERAGE")
    return np.asarray(
        [str(lookup[int(value)]) for value in household_ids], dtype=object
    )


class _PreparedKernel(KernelBase):
    """Every kernel of this slice shares one reviewed implementation scope."""

    def implementation_hash(self) -> str:
        return implementation_hash(ASEC_PREPARED_STAGE)


class USAsecPreparedCreateKernel(_PreparedKernel):
    """Prepare the whole authenticated ASEC population from one source directory."""

    ref = "us.asec_prepared.create@3"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.CREATE,
        dependencies=ASEC_PREPARED_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context)
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(tuple(node.sources) == (ASEC_PREPARED_SOURCE_NAME,), "NODE_SOURCES")
        _require(not node.inputs and not node.artifact_inputs, "NODE_INPUTS")
        _require(
            node.artifact_outputs
            == (
                ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ArtifactOutput("current_money", US_ASEC_CURRENT_MONEY_BODY_TYPE),
                ArtifactOutput("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
                ArtifactOutput("housing_universe", US_ASEC_HOUSING_UNIVERSE_TYPE),
                ArtifactOutput("income_observations", US_ASEC_INCOME_OBSERVATIONS_TYPE),
            ),
            "NODE_ARTIFACT_OUTPUTS",
        )
        prepared = prepare_asec_current_money_population(
            context.sources[ASEC_PREPARED_SOURCE_NAME]
        )
        _require(
            frame_column_declarations(prepared.frame) == node.outputs,
            "PREPARED_COLUMN_INVENTORY",
        )
        receipt = prepared.receipt
        consumers = load_graph_current_money_consumers()
        _require(receipt["source_kind"] == PREPARED_SOURCE_KIND, "PREPARED_SOURCE_KIND")
        _require(receipt["file_roster"] == list(PREPARED_SOURCE_FILES), "FILE_ROSTER")
        return KernelResult(
            frame=prepared.frame,
            artifacts={
                "frame_context": graph_context.encode_us_frame_context(prepared.frame),
                "current_money": prepared.money_payload,
                "prepared_receipt": prepared.receipt_payload,
                "housing_universe": prepared.housing_universe_payload,
                "income_observations": prepared.income_observations_payload,
            },
            receipt={
                "phase": ASEC_PREPARED_PHASE,
                "implementation": implementation_manifest(ASEC_PREPARED_STAGE),
                "prepared": receipt,
                "current_money_consumers": consumers,
                "release_eligible": False,
            },
        )


class USAsecPreparedSelectionKernel(_PreparedKernel):
    """Draw the seeded whole-household sample and slice the money body to it."""

    ref = "us.asec_prepared.select_households@3"
    capabilities = Capabilities(
        determinism=Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.PARAM,
        structural=StructuralDelta.FILTER,
        dependencies=ASEC_PREPARED_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, ("fraction", "seed"))
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(not node.outputs and not node.sources, "NODE_OUTPUTS")
        _require(node.structural is StructuralDelta.FILTER, "NODE_STRUCTURAL")
        _require(node.mass == "declared", "NODE_MASS")
        _require(
            set(context.tables) == {"person", "household"}
            and set(context.artifacts)
            == {
                "frame_context",
                "current_money",
                "prepared_receipt",
                "housing_universe",
            },
            "NODE_INPUTS",
        )
        _require(
            node.inputs
            == (
                Slice("person", (SELECTION_STRATUM_COLUMN,)),
                Slice(
                    "household", (HOUSEHOLD_WEIGHT_CARRIER, *HOUSEHOLD_EVIDENCE_COLUMNS)
                ),
            ),
            "NODE_SLICES",
        )
        _require(
            node.artifact_outputs
            == (
                ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ArtifactOutput("selection", US_ASEC_SELECTION_TYPE),
                ArtifactOutput("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
            ),
            "NODE_ARTIFACT_OUTPUTS",
        )
        bound = _artifact(context, "frame_context", US_FRAME_CONTEXT_TYPE)
        money = _artifact(context, "current_money", US_ASEC_CURRENT_MONEY_BODY_TYPE)
        receipt_value = _artifact(
            context, "prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE
        )
        producer_key = _same_producer(
            context,
            ("frame_context", "current_money", "prepared_receipt", "housing_universe"),
        )
        document = _document(bound)
        person, household = context.tables["person"], context.tables["household"]
        for entity, table in (("person", person), ("household", household)):
            _check_identity(document, entity, table)
        _require(
            document["weight_sources"] == {"household": "design"}, "WEIGHT_AUTHORITY"
        )
        _require(
            context.weights["household"].kind.value == "design", "WEIGHT_AUTHORITY"
        )
        prepared = json.loads(receipt_value.payload)
        _require(
            canonical_json(prepared) == receipt_value.payload, "PREPARED_RECEIPT_JSON"
        )
        _require(
            prepared["source_kind"] == PREPARED_SOURCE_KIND
            and prepared["release_eligible"] is False,
            "PREPARED_RECEIPT_SCHEMA",
        )
        _require(
            prepared["entity_rows"]["person"] == document["entities"]["person"]["rows"]
            and prepared["entity_rows"]["household"]
            == document["entities"]["household"]["rows"],
            "PREPARED_RECEIPT_ROWS",
        )
        body = parse_current_money_body(
            money.payload,
            expected_header_sha256=prepared["money_header_sha256"],
            expected_content_sha256=prepared["money_content_sha256"],
            field_entities=tuple(tuple(item) for item in prepared["field_entities"]),
        )
        _require(
            body.person_rows == document["entities"]["person"]["rows"]
            and body.household_rows == document["entities"]["household"]["rows"],
            "BODY_ROW_ALIGNMENT",
        )
        household_ids = household[US_SCHEMA.entity_id_column("household")].to_numpy(
            dtype="int64"
        )
        strata = _household_strata(person, household_ids)
        universe_value = _artifact(
            context, "housing_universe", US_ASEC_HOUSING_UNIVERSE_TYPE
        )
        universe = bind_housing_universe(
            universe_value.payload, prepared_receipt=prepared
        )
        verify_graph_housing_rows(
            household,
            universe,
            positions=np.arange(len(household), dtype="int64"),
            income_years=strata.astype("int64"),
        )
        view = _minimal_frame(
            person, household, context.weights["household"], context.strata
        )
        sampled, selection = sample_frame_households(
            view,
            fraction=context.params["fraction"],
            seed=context.params["seed"],
            source_name=SELECTION_SOURCE_LABEL,
            unit_strata=strata,
            unit_noun="household",
            floor_context="the ASEC prepared current-money engineering sample",
        )
        selected_household_ids = sampled.table("household")[
            US_SCHEMA.entity_id_column("household")
        ].to_numpy(dtype="int64")
        membership = person[US_SCHEMA.membership_column("household")].to_numpy(
            dtype="int64"
        )
        keep = np.isin(membership, selected_household_ids)
        person_positions = np.flatnonzero(keep).astype("int64")
        household_positions = np.flatnonzero(
            np.isin(household_ids, selected_household_ids)
        ).astype("int64")
        _require(len(person_positions) > 0, "EMPTY_SELECTION")
        _require(
            np.array_equal(household_ids[household_positions], selected_household_ids),
            "HOUSEHOLD_POSITIONS",
        )
        # Whole households only: no selected household may leave a member behind.
        _require(
            np.array_equal(
                person[_PERSON_ID].to_numpy(dtype="int64")[keep],
                sampled.person[_PERSON_ID].to_numpy(dtype="int64"),
            ),
            "WHOLE_HOUSEHOLD",
        )
        parent_weights = np.asarray(
            context.weights["household"].values, dtype="float64"
        )
        selected_weights = np.asarray(
            sampled.weights_for("household").values, dtype="float64"
        )
        _require(
            selected_weights.tobytes() == parent_weights[household_positions].tobytes(),
            "SELECTED_WEIGHT_BYTES",
        )
        entities = {"person": person[_PERSON_ID].to_numpy(dtype="int64")[keep]}
        for group in US_SCHEMA.group_entities:
            entities[group] = _group_ids(person.loc[keep], group)
        _require(
            np.array_equal(entities["household"], selected_household_ids),
            "SELECTED_HOUSEHOLD_INVENTORY",
        )
        identities = {
            entity: _identity(ids, entity) for entity, ids in entities.items()
        }
        before_mass = view.stratum_mass()
        after_mass = sampled.stratum_mass()
        payload = canonical_json(
            {
                "schema": SELECTION_SCHEMA,
                "phase": ASEC_PREPARED_PHASE,
                "release_eligible": False,
                "representative": False,
                "claim": "engineering_sample_only_no_district_or_calibration_claim",
                "stratum_column": SELECTION_STRATUM_COLUMN,
                "stratum_grain": "household_source_year",
                "exact_count_rule": EXACT_COUNT_RULE,
                "mass_normalization": "none",
                "selection": _json(selection),
                "entities": {
                    entity: {
                        "rows": identities[entity]["rows"],
                        "ordered_ids_sha256": identities[entity]["ordered_ids_sha256"],
                    }
                    for entity in _ENTITIES
                },
                "person_positions_sha256": _sha(person_positions.tobytes()),
                "household_positions_sha256": _sha(household_positions.tobytes()),
                "parent": {
                    "producer_key": producer_key,
                    "frame_context_sha256": _sha(bound.payload),
                    "prepared_receipt_sha256": _sha(receipt_value.payload),
                    "money_header_sha256": prepared["money_header_sha256"],
                    "money_content_sha256": prepared["money_content_sha256"],
                    "housing_universe_payload_sha256": _sha(universe_value.payload),
                },
                "mass": {
                    "policy": "declared",
                    "before": float(before_mass.sum()),
                    "after": float(after_mass.sum()),
                },
            }
        )
        selected = select_current_money(
            body,
            person_positions=person_positions,
            household_positions=household_positions,
            prepared_receipt_sha256=_sha(receipt_value.payload),
            selection_sha256=_sha(payload),
            person_identity_sha256=identities["person"]["ordered_ids_sha256"],
            household_identity_sha256=identities["household"]["ordered_ids_sha256"],
        )
        for entity in _ENTITIES:
            document["entities"][entity].update(identities[entity])
        return KernelResult(
            keep=pd.Series(
                keep,
                index=pd.Index(person[_PERSON_ID].to_numpy(), name=_PERSON_ID),
                dtype=bool,
            ),
            artifacts={
                "frame_context": canonical_json(document),
                "selection": payload,
                "selected_current_money": encode_selected_current_money(selected),
            },
            receipt={
                "phase": ASEC_PREPARED_PHASE,
                "implementation": implementation_manifest(ASEC_PREPARED_STAGE),
                "selection_sha256": _sha(payload),
                "selected_money_sha256": _sha(encode_selected_current_money(selected)),
                "release_eligible": False,
                "mass": {
                    "policy": "declared",
                    "before": float(before_mass.sum()),
                    "after": float(after_mass.sum()),
                    "stratum_before": {
                        str(key): float(value) for key, value in before_mass.items()
                    },
                    "stratum_after": {
                        str(key): float(value) for key, value in after_mass.items()
                    },
                },
            },
        )


def _json(value: object) -> object:
    """Normalize a receipt for canonical JSON without inventing string casts."""
    return graph_context._json_data(value)


class USAsecPreparedLeavesKernel(_PreparedKernel):
    """Derive the corrected monetary CPS leaves from the selected money alone."""

    ref = "us.asec_prepared.cps_carried_current@3"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=ASEC_PREPARED_DEPENDENCIES,
    )

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context)
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(not node.sources, "NODE_SOURCES")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        _require(
            node.inputs
            == (
                Slice("person", CPS_CARRIED_CURRENT_ROUTING_COLUMNS),
                Slice("household", HOUSEHOLD_EVIDENCE_COLUMNS),
            ),
            "NODE_SLICES",
        )
        _require(node.outputs == owned_leaf_declarations(), "NODE_OUTPUTS")
        _require(
            set(context.artifacts)
            == {
                "frame_context",
                "selection",
                "selected_current_money",
                "prepared_receipt",
                "housing_universe",
            },
            "NODE_ARTIFACT_INPUTS",
        )
        _require(
            node.artifact_outputs
            == (ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
            "NODE_ARTIFACT_OUTPUTS",
        )
        bound = _artifact(context, "frame_context", US_FRAME_CONTEXT_TYPE)
        selection_value = _artifact(context, "selection", US_ASEC_SELECTION_TYPE)
        money = _artifact(
            context, "selected_current_money", US_ASEC_SELECTED_MONEY_TYPE
        )
        receipt_value = _artifact(
            context, "prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE
        )
        _same_producer(
            context, ("frame_context", "selection", "selected_current_money")
        )
        universe_value = _artifact(
            context, "housing_universe", US_ASEC_HOUSING_UNIVERSE_TYPE
        )
        _same_producer(context, ("prepared_receipt", "housing_universe"))
        document = _document(bound)
        person, spm_unit = context.tables["person"], context.tables["spm_unit"]
        for entity, table in (("person", person), ("spm_unit", spm_unit)):
            _check_identity(document, entity, table)
        selection = json.loads(selection_value.payload)
        _require(canonical_json(selection) == selection_value.payload, "SELECTION_JSON")
        _require(selection["schema"] == SELECTION_SCHEMA, "SELECTION_SCHEMA")
        prepared = json.loads(receipt_value.payload)
        _require(
            selection["parent"]["prepared_receipt_sha256"]
            == _sha(receipt_value.payload)
            and selection["parent"]["producer_key"] == receipt_value.producer_key,
            "SELECTION_PARENT_BINDING",
        )
        _require(
            selection["parent"]["housing_universe_payload_sha256"]
            == _sha(universe_value.payload),
            "SELECTION_HOUSING_UNIVERSE_BINDING",
        )
        universe = bind_housing_universe(
            universe_value.payload, prepared_receipt=prepared
        )
        household = context.tables["household"]
        _check_identity(document, "household", household)
        positions = (
            pd.Index(universe.array("household_id"))
            .get_indexer(household.household_id.to_numpy())
            .astype("int64")
        )
        _require(
            _sha(positions.tobytes()) == selection["household_positions_sha256"],
            "SELECTED_HOUSING_UNIVERSE_POSITIONS",
        )
        verify_graph_housing_rows(household, universe, positions=positions)
        # The complete selected-coordinate check: the selection artifact, the
        # producer's typed context and the actual rows must name one row set.
        for entity in _ENTITIES:
            declared = document["entities"][entity]
            _require(
                selection["entities"][entity]["rows"] == declared["rows"]
                and selection["entities"][entity]["ordered_ids_sha256"]
                == declared["ordered_ids_sha256"],
                f"SELECTION_IDENTITY:{entity}",
            )
        selected = decode_selected_current_money(
            money.payload,
            expected_parent_header_sha256=prepared["money_header_sha256"],
            expected_parent_content_sha256=prepared["money_content_sha256"],
            expected_prepared_receipt_sha256=_sha(receipt_value.payload),
            expected_selection_sha256=_sha(selection_value.payload),
        )
        header = selected.header_data
        _require(
            header["person_identity_sha256"]
            == document["entities"]["person"]["ordered_ids_sha256"]
            and header["household_identity_sha256"]
            == document["entities"]["household"]["ordered_ids_sha256"],
            "SELECTED_COORDINATES",
        )
        _require(
            selected.person_rows == len(person)
            and selected.household_rows == document["entities"]["household"]["rows"],
            "SELECTED_ROW_ALIGNMENT",
        )
        incumbents = sorted(
            f"{entity}.{column}"
            for entity, column in _owned_coordinates()
            if column in document["entities"][entity]["columns"]
        )
        _require(not incumbents, "OWNED_LEAF_INCUMBENT")
        routing = person.loc[:, list(CPS_CARRIED_CURRENT_ROUTING_COLUMNS)]
        spm_ids = spm_unit[US_SCHEMA.entity_id_column("spm_unit")].to_numpy(
            dtype="int64"
        )
        leaves = derive_cps_carried_current_leaves(
            selected,
            routing=routing,
            spm_membership=person[US_SCHEMA.membership_column("spm_unit")].to_numpy(
                dtype="int64"
            ),
            spm_ids=spm_ids,
        )
        person_index = pd.Index(person[_PERSON_ID].to_numpy(), name=_PERSON_ID)
        spm_index = pd.Index(spm_ids, name=US_SCHEMA.entity_id_column("spm_unit"))
        columns = {
            ("person", name): pd.Series(values, index=person_index, dtype=LEAF_DTYPE)
            for name, values in leaves.person.items()
        }
        columns.update(
            {
                ("spm_unit", name): pd.Series(values, index=spm_index, dtype=LEAF_DTYPE)
                for name, values in leaves.spm_unit.items()
            }
        )
        for entity, column in _owned_coordinates():
            document["entities"][entity]["columns"].append(column)
        return KernelResult(
            columns=columns,
            artifacts={"frame_context": canonical_json(document)},
            receipt={
                "phase": ASEC_PREPARED_PHASE,
                "implementation": implementation_manifest(ASEC_PREPARED_STAGE),
                "contract": cps_carried_current_leaf_contract(),
                "selected_money_sha256": _sha(money.payload),
                "selection_sha256": _sha(selection_value.payload),
                "person_rows": len(person),
                "spm_unit_rows": len(spm_ids),
                "release_eligible": False,
            },
        )


class USAsecPreparedEngineKernel(_PreparedKernel):
    """Run the real PolicyEngine-US adapter for the admitted formula-owned roots."""

    ref = "us.asec_prepared.engine_current_money@3"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=(
            *ASEC_PREPARED_DEPENDENCIES,
            "policyengine-us",
            "policyengine-core",
        ),
    )

    def implementation_hash(self) -> str:
        # The executor checks this identity before cache lookup. Runtime or
        # baseline parameter drift therefore refuses even when run() is unused.
        return _sha(
            canonical_json(
                {
                    "implementation": super().implementation_hash(),
                    "runtime": engine_runtime_identity(),
                }
            )
        )

    def __init__(self, engine=None) -> None:
        # Only a real adapter instance is accepted; a substitute would make the
        # evaluation artifact a claim about something other than the engine.
        if engine is not None and type(engine) is not PolicyEngineUSEngine:
            raise PreparedGraphError("ENGINE_ADAPTER_TYPE")
        self._engine = engine

    def run(self, context: KernelContext) -> KernelResult:
        _phase(context, ("period", "engine_version"))
        node = context.node
        _require(node.kernel == self.ref, "NODE_KERNEL")
        _require(not node.sources, "NODE_SOURCES")
        _require(node.structural is StructuralDelta.NONE, "NODE_STRUCTURAL")
        # Engine outputs are formula-owned; this node owns no population cell.
        _require(not node.outputs, "ENGINE_OWNS_NO_CELLS")
        _require(
            node.inputs
            == (
                Slice("person", CPS_CARRIED_CURRENT_PERSON_LEAVES),
                Slice("spm_unit", CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES),
                Slice("household", (HOUSEHOLD_WEIGHT_CARRIER,)),
            ),
            "NODE_SLICES",
        )
        _require(set(context.artifacts) == {"frame_context"}, "NODE_ARTIFACT_INPUTS")
        _require(
            node.artifact_outputs
            == (ArtifactOutput("engine_evaluation", US_ASEC_ENGINE_EVALUATION_TYPE),),
            "NODE_ARTIFACT_OUTPUTS",
        )
        bound = _artifact(context, "frame_context", US_FRAME_CONTEXT_TYPE)
        document = _document(bound)
        for entity in ("person", "spm_unit", "household"):
            _check_identity(document, entity, context.tables[entity])
        for entity, column in _owned_coordinates():
            _require(
                column in document["entities"][entity]["columns"],
                f"LEAF_NOT_IN_CONTEXT:{entity}.{column}",
            )
        frame = self._engine_frame(context, document)
        index = PolicyEngineUSVariableMetadataIndex()
        contracts = admit_engine_outputs(
            index, produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES
        )
        _require(
            all(
                contract.engine_version == context.params["engine_version"]
                for contract in contracts
            ),
            "DECLARED_ENGINE_VERSION",
        )
        evaluation = materialize_engine_outputs(
            frame,
            PolicyEngineUSEngine() if self._engine is None else self._engine,
            contracts,
            period=context.params["period"],
            context_bindings={
                "frame_context_sha256": _sha(bound.payload),
                "frame_context_producer_key": bound.producer_key,
                "produced_leaves": list(CPS_CARRIED_CURRENT_PERSON_LEAVES),
                "spm_unit_leaves": list(CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES),
            },
        )
        payload = encode_engine_evaluation(evaluation)
        return KernelResult(
            artifacts={"engine_evaluation": payload},
            receipt={
                "phase": ASEC_PREPARED_PHASE,
                "implementation": implementation_manifest(ASEC_PREPARED_STAGE),
                "engine": {
                    "package": evaluation.header["engine_package"],
                    "version": evaluation.header["engine_version"],
                    "period": evaluation.header["period"],
                    "roots": list(ADMITTED_ROOTS),
                    "blocked_roots": list(BLOCKED_ENGINE_OUTPUTS),
                    "closures": evaluation.header["closures"],
                    "recorded_defaults": [],
                    "aggregates": evaluation.header["aggregates"],
                },
                "evaluation_sha256": _sha(payload),
                "cells_written": 0,
                "release_eligible": False,
                "claim": "engineering_evidence_only_no_tax_or_benefit_score",
            },
        )

    def _engine_frame(self, context: KernelContext, document: dict) -> Frame:
        """Rebuild the selected frame carrying only leaves and id carriers.

        Every raw ASEC column, every routing code and the household carrier are
        left behind: what the engine receives is exactly the corrected leaves
        plus the structure they are indexed by.
        """
        person = context.tables["person"]
        keep = [
            _PERSON_ID,
            *(US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities),
            *CPS_CARRIED_CURRENT_PERSON_LEAVES,
        ]
        _require(set(person.columns) == set(keep), "ENGINE_PERSON_COLUMNS")
        tables = {"person": person.loc[:, keep].copy(deep=True)}
        for group in US_SCHEMA.group_entities:
            ids = _group_ids(person, group)
            declared = document["entities"][group]
            actual = _identity(ids, group)
            _require(
                all(declared[key] == value for key, value in actual.items()),
                f"ENGINE_GROUP_IDENTITY:{group}",
            )
            tables[group] = pd.DataFrame({US_SCHEMA.entity_id_column(group): ids})
        for group in ("household", "spm_unit"):
            view = context.tables[group]
            _require(
                np.array_equal(
                    view[US_SCHEMA.entity_id_column(group)].to_numpy(dtype="int64"),
                    tables[group][US_SCHEMA.entity_id_column(group)].to_numpy(),
                ),
                f"ENGINE_GROUP_VIEW:{group}",
            )
        spm = context.tables["spm_unit"]
        for name in CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES:
            tables["spm_unit"][name] = spm[name].to_numpy(dtype="float64", copy=True)
        weights = context.weights["household"]
        _require(weights.kind.value == "design", "ENGINE_WEIGHT_AUTHORITY")
        return Frame(
            tables,
            US_SCHEMA,
            {
                "household": Weights(
                    np.asarray(weights.values, dtype="float64"), weights.kind
                )
            },
        )


def _owned_coordinates() -> tuple[tuple[str, str], ...]:
    return (
        *(("person", name) for name in CPS_CARRIED_CURRENT_PERSON_LEAVES),
        *(("spm_unit", name) for name in CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES),
    )


def owned_leaf_declarations() -> tuple[Owned, ...]:
    """The cells the corrected-leaves node owns, in one canonical order."""
    return tuple(
        Owned(entity, column, LEAF_DTYPE) for entity, column in _owned_coordinates()
    )


def us_asec_prepared_graph(
    columns: Sequence[Owned],
    *,
    fraction: float,
    seed: int,
    period: int = 2024,
    engine_version: str = "1.819.0",
) -> Graph:
    """Declare the five-node slice over the caller's prepared column inventory."""
    inventory = {(owned.entity, owned.column) for owned in columns}
    _require(len(inventory) == len(tuple(columns)), "COLUMN_INVENTORY_REPEATS")
    _require(
        ("household", HOUSEHOLD_WEIGHT_CARRIER) in inventory, "MISSING_WEIGHT_CARRIER"
    )
    missing = sorted(
        name
        for name in (SELECTION_STRATUM_COLUMN, *CPS_CARRIED_CURRENT_ROUTING_COLUMNS)
        if ("person", name) not in inventory
    )
    _require(not missing, f"MISSING_DECLARED_INPUTS:{missing}")
    _require(
        all(("household", name) in inventory for name in HOUSEHOLD_EVIDENCE_COLUMNS),
        "MISSING_HOUSING_UNIVERSE_INPUTS",
    )
    # A nominal incumbent can be neither preserved nor silently rewritten.
    incumbent = sorted(
        f"{entity}.{column}"
        for entity, column in _owned_coordinates()
        if (entity, column) in inventory
    )
    incumbent.extend(
        f"person.{name}" for name in RESULT_COLUMNS if ("person", name) in inventory
    )
    _require(not incumbent, f"OWNED_LEAF_INCUMBENT:{incumbent}")
    parent_artifacts = (
        ArtifactInput(
            "frame_context", CREATE_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
        ),
        ArtifactInput(
            "current_money",
            CREATE_NODE,
            "current_money",
            US_ASEC_CURRENT_MONEY_BODY_TYPE,
        ),
        ArtifactInput(
            "prepared_receipt",
            CREATE_NODE,
            "prepared_receipt",
            US_ASEC_PREPARED_RECEIPT_TYPE,
        ),
        ArtifactInput(
            "housing_universe",
            CREATE_NODE,
            "housing_universe",
            US_ASEC_HOUSING_UNIVERSE_TYPE,
        ),
    )
    create = Node(
        id=CREATE_NODE,
        kernel=USAsecPreparedCreateKernel.ref,
        structural=StructuralDelta.CREATE,
        sources=(ASEC_PREPARED_SOURCE_NAME,),
        outputs=tuple(columns),
        params={"phase": ASEC_PREPARED_PHASE},
        artifact_outputs=(
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ArtifactOutput("current_money", US_ASEC_CURRENT_MONEY_BODY_TYPE),
            ArtifactOutput("prepared_receipt", US_ASEC_PREPARED_RECEIPT_TYPE),
            ArtifactOutput("housing_universe", US_ASEC_HOUSING_UNIVERSE_TYPE),
            ArtifactOutput("income_observations", US_ASEC_INCOME_OBSERVATIONS_TYPE),
        ),
    )
    select = Node(
        id=SELECT_NODE,
        kernel=USAsecPreparedSelectionKernel.ref,
        base=CREATE_NODE,
        structural=StructuralDelta.FILTER,
        mass="declared",
        inputs=(
            Slice("person", (SELECTION_STRATUM_COLUMN,)),
            Slice("household", (HOUSEHOLD_WEIGHT_CARRIER, *HOUSEHOLD_EVIDENCE_COLUMNS)),
        ),
        params={"phase": ASEC_PREPARED_PHASE, "fraction": fraction, "seed": seed},
        artifact_inputs=parent_artifacts,
        artifact_outputs=(
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ArtifactOutput("selection", US_ASEC_SELECTION_TYPE),
            ArtifactOutput("selected_current_money", US_ASEC_SELECTED_MONEY_TYPE),
        ),
    )
    leaves = Node(
        id=LEAVES_NODE,
        kernel=USAsecPreparedLeavesKernel.ref,
        population=SELECT_NODE,
        inputs=(
            Slice("person", CPS_CARRIED_CURRENT_ROUTING_COLUMNS),
            Slice("household", HOUSEHOLD_EVIDENCE_COLUMNS),
        ),
        outputs=owned_leaf_declarations(),
        params={"phase": ASEC_PREPARED_PHASE},
        artifact_inputs=(
            ArtifactInput(
                "frame_context", SELECT_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
            ),
            ArtifactInput(
                "selection", SELECT_NODE, "selection", US_ASEC_SELECTION_TYPE
            ),
            ArtifactInput(
                "selected_current_money",
                SELECT_NODE,
                "selected_current_money",
                US_ASEC_SELECTED_MONEY_TYPE,
            ),
            parent_artifacts[2],
            parent_artifacts[3],
        ),
        artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
    )
    engine = Node(
        id=ENGINE_NODE,
        kernel=USAsecPreparedEngineKernel.ref,
        population=SELECT_NODE,
        inputs=(
            Slice("person", CPS_CARRIED_CURRENT_PERSON_LEAVES),
            Slice("spm_unit", CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES),
            Slice("household", (HOUSEHOLD_WEIGHT_CARRIER,)),
        ),
        params={
            "phase": ASEC_PREPARED_PHASE,
            "period": period,
            "engine_version": engine_version,
        },
        artifact_inputs=(
            ArtifactInput(
                "frame_context", LEAVES_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
            ),
        ),
        artifact_outputs=(
            ArtifactOutput("engine_evaluation", US_ASEC_ENGINE_EVALUATION_TYPE),
        ),
    )
    reported = Node(
        id=REPORTED_INCOME_NODE,
        kernel=USAsecReportedIncomeKernel.ref,
        population=SELECT_NODE,
        inputs=(Slice("person", ("source_year",)),),
        outputs=reported_income_declarations(),
        params={"phase": ASEC_PREPARED_PHASE},
        artifact_inputs=(
            *leaves.artifact_inputs[:3],
            parent_artifacts[2],
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
    return Graph(
        country="us",
        sources=(ASEC_PREPARED_SOURCE,),
        nodes=(create, select, leaves, engine, reported),
    )


def us_asec_prepared_registry(engine=None) -> KernelRegistry:
    """Register exactly this slice's five kernels."""
    registry = KernelRegistry()
    registry.register(USAsecPreparedCreateKernel())
    registry.register(USAsecPreparedSelectionKernel())
    registry.register(USAsecPreparedLeavesKernel())
    registry.register(USAsecPreparedEngineKernel(engine))
    registry.register(USAsecReportedIncomeKernel())
    return registry


__all__ = [
    "ASEC_PREPARED_CODEC",
    "ASEC_PREPARED_DEPENDENCIES",
    "ASEC_PREPARED_PHASE",
    "ASEC_PREPARED_SOURCE",
    "ASEC_PREPARED_SOURCE_NAME",
    "ASEC_PREPARED_STAGE",
    "CREATE_NODE",
    "ENGINE_NODE",
    "HOUSEHOLD_WEIGHT_CARRIER",
    "LEAVES_NODE",
    "PreparedGraphError",
    "SELECTION_SCHEMA",
    "SELECT_NODE",
    "USAsecPreparedCreateKernel",
    "USAsecPreparedEngineKernel",
    "USAsecPreparedLeavesKernel",
    "USAsecPreparedSelectionKernel",
    "owned_leaf_declarations",
    "us_asec_prepared_graph",
    "us_asec_prepared_registry",
]
