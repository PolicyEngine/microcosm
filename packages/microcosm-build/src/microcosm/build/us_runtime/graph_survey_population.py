"""Graph declarations for a source-authenticated, selected survey population.

The catalogue draw precedes native construction. CREATE therefore materializes
the selected combined population, and a separate REWEIGHT applies source shares
and inverse inclusion. Neither decoded evidence nor these declarations grants
source, population, or release authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.frame import US_SCHEMA, MassChange, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Ownership,
    Slice,
    StructuralDelta,
    WeightTransition,
)

from . import graph_context
from .graph_context import US_FRAME_CONTEXT_TYPE
from .support_provenance import (
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from .survey_catalogue_selection import CatalogueSelectionPlan, SelectedHousehold
from .survey_population_domains import HouseholdKey, Source

PHASE = "us.authenticated_survey_population.v1"
SOURCE_NAME = "survey_population_source"
SOURCE_CODEC = "us-survey-population-source-v1"
CREATE_NODE = "survey_population.create"
ALLOCATION_NODE = "survey_population.allocate"
CREATE_REF = "us.survey_population.create@1"
ALLOCATION_REF = "us.survey_population.allocate@1"
PREPARATION_TYPE = ArtifactType("microcosm.us.survey_population_preparation", 2)
ALLOCATION_TYPE = ArtifactType("microcosm.us.survey_population_allocation", 1)
PREPARATION_MAX_BYTES = 64 * 1024**2
ALLOCATION_MAX_BYTES = 64 * 1024**2
STAGE = "authenticated_survey_population_v1"
STAGE_DEPENDENCIES = (
    "numpy",
    "pandas",
    "microunit",
    "tables",
    "h5py",
    "PyYAML",
    "pyarrow",
)
_HOUSEHOLD_ORIGIN_FIELDS = frozenset(
    {
        "household_id",
        "source",
        "source_year",
        "survey_year",
        "raw_native_id",
        "selected_receiving_household_id",
        "original_anchor",
    }
)


@dataclass(frozen=True, slots=True)
class AllocationInstruction:
    """Exact arithmetic values, without source or population authority."""

    household_id: int
    key: HouseholdKey
    selected_receiving_household_id: int
    original_anchor: Fraction
    share: Fraction
    inclusion_probability: Fraction

    @property
    def importance_weight(self):
        return self.original_anchor * self.share / self.inclusion_probability


class SurveyPopulationGraphError(ValueError):
    """Static graph refusal without source values or local paths."""


def _require(condition, code):
    if not condition:
        raise SurveyPopulationGraphError(code)


def _digest(value):
    _require(
        type(value) is str
        and len(value) == 64
        and not set(value) - set("0123456789abcdef"),
        "DIGEST",
    )
    return value


def _draw_parameters(fraction, seed):
    _require(type(fraction) is Fraction and 0 < fraction <= 1, "FRACTION")
    _require(type(seed) is int and 0 <= seed < 2**64, "SEED")
    return (fraction.numerator, fraction.denominator), seed


def _fraction_pair(value):
    _require(
        type(value) in (list, tuple)
        and len(value) == 2
        and all(type(v) is int for v in value)
        and value[1] > 0,
        "FRACTION_PAIR",
    )
    number = Fraction(*value)
    _require((number.numerator, number.denominator) == tuple(value), "FRACTION_PAIR")
    return number


def allocation_instructions(plan, household_origins):
    """Join checked preparation values; this pure join authenticates nothing.

    The runner obtains both arguments from a freshly issued preparation. Keeping
    this operation separate lets the allocation kernel hold immutable values
    instead of a source Frame or a mutable decoded preparation document.
    """
    _require(type(plan) is CatalogueSelectionPlan, "SELECTION_PLAN")
    _require(type(plan.selected) is tuple and plan.selected, "EMPTY_SELECTION")
    _require(type(household_origins) in (list, tuple), "HOUSEHOLD_ORIGINS")
    _require(len(household_origins) == len(plan.selected), "ORIGIN_COUNT")
    # A prospective record bound precedes the two maps and output tuple. The
    # streaming transport encoder enforces its exact bound separately.
    _require(len(plan.selected) <= ALLOCATION_MAX_BYTES // 128, "ALLOCATION_LIMIT")
    selected = {}
    for row in plan.selected:
        _require(
            type(row) is SelectedHousehold and type(row.key) is HouseholdKey,
            "SELECTED_TYPE",
        )
        _require(
            type(row.key.source) is Source
            and type(row.key.source_year) is int
            and row.key.source_year == 2024
            and type(row.key.survey_year) is int
            and row.key.survey_year == (2024 if row.key.source is Source.ACS else 2025)
            and type(row.key.native_id) is str
            and 0 < len(row.key.native_id) <= 128,
            "SELECTED_KEY",
        )
        _require(row.key not in selected, "SELECTED_DUPLICATE")
        _require(
            type(row.original_design_weight) is Fraction
            and row.original_design_weight >= 0
            and type(row.share) is Fraction
            and 0 < row.share <= 1
            and type(row.inclusion_probability) is Fraction
            and 0 < row.inclusion_probability <= 1,
            "SELECTED_ARITHMETIC",
        )
        selected[row.key] = row
    output, seen_ids, seen_native, seen_receiving = [], set(), set(), set()
    for origin in household_origins:
        _require(
            type(origin) is dict and set(origin) == _HOUSEHOLD_ORIGIN_FIELDS,
            "ORIGIN_FIELDS",
        )
        _require(
            type(origin["source"]) is str and origin["source"] in {"acs", "asec"},
            "ORIGIN_SOURCE",
        )
        _require(
            all(
                type(origin[field]) is int
                for field in (
                    "household_id",
                    "source_year",
                    "survey_year",
                    "selected_receiving_household_id",
                )
            ),
            "ORIGIN_INTEGER",
        )
        _require(
            all(
                0 <= origin[field] < 2**63
                for field in ("household_id", "selected_receiving_household_id")
            ),
            "ORIGIN_INTEGER",
        )
        _require(type(origin["raw_native_id"]) is str, "ORIGIN_NATIVE_KEY")
        key = HouseholdKey(
            Source(origin["source"]),
            origin["source_year"],
            origin["survey_year"],
            origin["raw_native_id"],
        )
        row = selected.get(key)
        _require(row is not None, "ORIGIN_NATIVE_KEY")
        anchor = _fraction_pair(origin["original_anchor"])
        _require(anchor == row.original_design_weight, "ORIGIN_ANCHOR")
        hh_id = origin["household_id"]
        receiving = (key.source, origin["selected_receiving_household_id"])
        _require(
            hh_id not in seen_ids
            and key not in seen_native
            and receiving not in seen_receiving,
            "ORIGIN_DUPLICATE",
        )
        seen_ids.add(hh_id)
        seen_native.add(key)
        seen_receiving.add(receiving)
        instruction = AllocationInstruction(
            hh_id, key, receiving[1], anchor, row.share, row.inclusion_probability
        )
        _finite_weight(anchor)
        _finite_weight(instruction.importance_weight)
        output.append(instruction)
    return tuple(output)


def _finite_weight(value):
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise SurveyPopulationGraphError("FLOAT_WEIGHT") from None
    _require(
        math.isfinite(number) and number >= 0 and (value == 0 or number > 0),
        "FLOAT_WEIGHT",
    )
    return number


def _bounded_json(value, limit):
    """Canonical UTF-8 encoding with a bound checked before every append."""
    _require(type(limit) is int and 0 < limit <= 64 * 1024**2, "TRANSPORT_LIMIT")
    result = bytearray()
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    try:
        for piece in encoder.iterencode(value):
            # All variable source tokens admitted to allocation are <=128 chars;
            # this pre-encoding bound also protects against unexpected strings.
            _require(len(piece) <= limit - len(result), "TRANSPORT_LIMIT")
            encoded = piece.encode("utf-8")
            _require(len(encoded) <= limit - len(result), "TRANSPORT_LIMIT")
            result.extend(encoded)
    except SurveyPopulationGraphError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise SurveyPopulationGraphError("TRANSPORT_ENCODING") from None
    return bytes(result)


def _source_owner():
    # The owner remains independent of this graph adapter. Declaration-only
    # imports do not load source capture owners or their runtime resources.
    from . import survey_population_preparation

    return survey_population_preparation


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _checked_preparation(preparation):
    owner = _source_owner()
    _require(
        type(preparation) is owner.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    view = owner.AuthenticatedSurveyPopulationPreparation.checked_view(preparation)
    payload, context = view.payload, view.context
    _require(
        type(payload) is bytes and 0 < len(payload) <= PREPARATION_MAX_BYTES,
        "PREPARATION_BYTES",
    )
    _require(
        type(context) is bytes and 0 < len(context) <= PREPARATION_MAX_BYTES,
        "CONTEXT_BYTES",
    )
    return owner, view


def _artifact(context, name, expected_type, expected_payload):
    from microcosm.graph.keys import opaque_artifact_key

    value = context.artifacts.get(name)
    edges = [edge for edge in context.node.artifact_inputs if edge.name == name]
    _require(
        value is not None
        and value.type == expected_type
        and len(edges) == 1
        and edges[0].producer == CREATE_NODE
        and edges[0].artifact == name
        and value.key == opaque_artifact_key(value.producer_key, name)
        and type(value.payload) is bytes
        and value.payload == expected_payload,
        "ARTIFACT_BINDING",
    )
    return value


def _instruction_document(row):
    def pair(value):
        return [value.numerator, value.denominator]

    return {
        "household_id": row.household_id,
        "source": row.key.source.value,
        "source_year": row.key.source_year,
        "survey_year": row.key.survey_year,
        "raw_native_id": row.key.native_id,
        "selected_receiving_household_id": row.selected_receiving_household_id,
        "original_anchor": pair(row.original_anchor),
        "share": pair(row.share),
        "inclusion_probability": pair(row.inclusion_probability),
        "importance_multiplier": pair(row.share / row.inclusion_probability),
        "importance_weight_float64_hex": _finite_weight(row.importance_weight).hex(),
    }


def _allocation_payload(
    instructions, *, preparation_sha256, input_context, output_context
):
    """Stream rows individually; never construct a selected-population JSON tree."""
    metadata = {
        "protocol": "microcosm.us.survey-population-allocation.v1",
        "preparation_sha256": preparation_sha256,
        "input_context_sha256": _sha(input_context),
        "output_context_sha256": _sha(output_context),
        "weight_transition": ["design", "importance"],
        "release_eligible": False,
    }
    tail = b"]," + _bounded_json(metadata, 4096)[1:]
    output = bytearray(b'{"households":[')
    for index, row in enumerate(instructions):
        raw = _bounded_json(_instruction_document(row), 4096)
        separator = b"," if index else b""
        _require(
            len(output) + len(separator) + len(raw) + len(tail) <= ALLOCATION_MAX_BYTES,
            "ALLOCATION_LIMIT",
        )
        output.extend(separator)
        output.extend(raw)
    _require(len(output) + len(tail) <= ALLOCATION_MAX_BYTES, "ALLOCATION_LIMIT")
    output.extend(tail)
    return bytes(output)


def _verify_allocation_view(frame, instructions):
    _require(frame.weighted_entities == ("household",), "WEIGHT_SOURCES")
    weights = frame.weights_for("household")
    _require(weights.kind is WeightKind.DESIGN, "DESIGN_REQUIRED")
    household = frame.table("household")
    _require(len(household) == len(instructions), "HOUSEHOLD_COUNT")
    _require(household.household_id.dtype == np.dtype("int64"), "HOUSEHOLD_ID_DTYPE")
    _require(
        tuple(household.household_id) == tuple(r.household_id for r in instructions),
        "HOUSEHOLD_ORDER",
    )
    _require(
        tuple(household[support_channel_column("household")])
        == tuple(r.key.source.value for r in instructions),
        "HOUSEHOLD_CHANNEL",
    )
    _require(
        tuple(household[spine_source_id_column("household")])
        == tuple(r.selected_receiving_household_id for r in instructions),
        "RECEIVING_ID",
    )
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        _require(
            table[support_clone_index_column(entity)].dtype == np.dtype("int64")
            and bool(table[support_clone_index_column(entity)].eq(0).all()),
            "PRECLONE_REQUIRED",
        )
    expected = np.asarray(
        [_finite_weight(r.original_anchor) for r in instructions], dtype="float64"
    )
    _require(
        weights.values.dtype == expected.dtype
        and weights.values.tobytes() == expected.tobytes(),
        "ORIGINAL_ANCHORS",
    )


def _allocation_output(frame, context_bytes, instructions, preparation_sha256):
    _verify_allocation_view(frame, instructions)
    values = np.asarray(
        [_finite_weight(r.importance_weight) for r in instructions], dtype="float64"
    )
    weights = Weights(values, WeightKind.IMPORTANCE)
    allocated = frame.with_weights(
        "household",
        weights,
        mass=MassChange(
            factor=None, reason=f"{PHASE}: declared source shares and inverse inclusion"
        ),
    )
    context = graph_context._decode(context_bytes)
    _require(
        context["weight_sources"] == {"household": "design"}, "CONTEXT_WEIGHT_KIND"
    )
    context["weight_sources"] = {"household": "importance"}
    context["mass_log"] = [
        graph_context._json_data(asdict(r)) for r in allocated.mass_log
    ]
    new_context = _bounded_json(context, PREPARATION_MAX_BYTES)
    payload = _allocation_payload(
        instructions,
        preparation_sha256=preparation_sha256,
        input_context=context_bytes,
        output_context=new_context,
    )
    before, after = frame.stratum_mass(), allocated.stratum_mass()
    receipt = {
        "phase": PHASE,
        "preparation_sha256": preparation_sha256,
        "allocation_sha256": _sha(payload),
        "release_eligible": False,
        "frame_mass_log_append": [
            graph_context._json_data(asdict(r))
            for r in allocated.mass_log[len(frame.mass_log) :]
        ],
        "mass": {
            "policy": "declared",
            "before": float(before.sum()),
            "after": float(after.sum()),
            "stratum_before": before.to_dict(),
            "stratum_after": after.to_dict(),
        },
    }
    return weights, new_context, payload, receipt, allocated


class _Kernel(KernelBase):
    def implementation_hash(self):
        from .graph_implementation import (
            STAGE_DEPENDENCIES as STAGES,
        )
        from .graph_implementation import (
            implementation_hash,
        )

        _require(STAGES[STAGE] == STAGE_DEPENDENCIES, "STAGE_DEPENDENCIES")
        return implementation_hash(STAGE)


class SurveyPopulationCreateKernel(_Kernel):
    ref = CREATE_REF
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.CREATE,
        dependencies=STAGE_DEPENDENCIES,
    )

    def __init__(self, preparation, *, source_dir):
        self._preparation = preparation
        self._source_dir = Path(source_dir).absolute()

    def run(self, context):
        from .graph_sources import frame_column_declarations

        preparation = self._preparation
        owner, view = _checked_preparation(preparation)
        payload, context_bytes, plan = view.payload, view.context, view.selection_plan
        expected, _ = survey_population_nodes(
            frame_column_declarations(view.frame),
            preparation_sha256=_sha(payload),
            fraction=plan.fraction,
            seed=plan.seed,
        )
        _require(
            context.node == expected and dict(context.params) == dict(expected.params),
            "CREATE_DECLARATION",
        )
        _require(
            set(context.sources) == {SOURCE_NAME}
            and Path(context.sources[SOURCE_NAME])
            == self._source_dir.resolve(strict=True),
            "SOURCE_PATH",
        )
        owner.verify_survey_population_preparation(preparation)
        return KernelResult(
            frame=view.frame,
            artifacts={"frame_context": context_bytes, "preparation": payload},
            receipt={
                "phase": PHASE,
                "preparation_sha256": _sha(payload),
                "source_reconstruction": "fresh_before_graph_execution",
                "release_eligible": False,
            },
        )


class SurveyPopulationAllocationKernel(_Kernel):
    ref = ALLOCATION_REF
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.REWEIGHT,
        dependencies=STAGE_DEPENDENCIES,
    )

    def __init__(self, instructions, *, preparation_bytes, context_bytes):
        _require(
            type(instructions) is tuple
            and instructions
            and all(type(row) is AllocationInstruction for row in instructions),
            "INSTRUCTIONS",
        )
        _require(
            type(preparation_bytes) is bytes
            and 0 < len(preparation_bytes) <= PREPARATION_MAX_BYTES,
            "PREPARATION_BYTES",
        )
        _require(
            type(context_bytes) is bytes
            and 0 < len(context_bytes) <= PREPARATION_MAX_BYTES,
            "CONTEXT_BYTES",
        )
        self._instructions = instructions
        self._preparation_bytes = preparation_bytes
        self._context_bytes = context_bytes

    def run(self, context):
        node = context.node
        digest = _sha(self._preparation_bytes)
        _require(
            node.id == ALLOCATION_NODE
            and node.kernel == self.ref
            and node.structural is StructuralDelta.REWEIGHT
            and node.base == CREATE_NODE
            and node.mass == "declared"
            and not node.outputs
            and not node.sources
            and node.weights
            == WeightTransition("household", "importance", mass="declared")
            and node.inputs
            == tuple(Slice(e, _provenance_columns(e)) for e in US_SCHEMA.entities)
            and node.artifact_inputs
            == (
                ArtifactInput(
                    "frame_context", CREATE_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
                ),
                ArtifactInput(
                    "preparation", CREATE_NODE, "preparation", PREPARATION_TYPE
                ),
            )
            and node.artifact_outputs
            == (
                ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ArtifactOutput("allocation", ALLOCATION_TYPE),
            )
            and dict(context.params) == {"phase": PHASE, "preparation_sha256": digest},
            "ALLOCATION_DECLARATION",
        )
        _require(
            set(context.artifacts) == {"frame_context", "preparation"}, "ARTIFACTS"
        )
        bound = _artifact(
            context, "frame_context", US_FRAME_CONTEXT_TYPE, self._context_bytes
        )
        prepared = _artifact(
            context, "preparation", PREPARATION_TYPE, self._preparation_bytes
        )
        _require(bound.producer_key == prepared.producer_key, "ARTIFACT_PRODUCER")
        original = graph_context.us_frame_from_context(context)
        weights, new_context, payload, receipt, _allocated = _allocation_output(
            original, bound.payload, self._instructions, digest
        )
        return KernelResult(
            weights=weights,
            artifacts={"frame_context": new_context, "allocation": payload},
            receipt=receipt,
        )


def survey_population_source_codecs(*, snapshot_root):
    """One real source codec, isolated from the legacy five-codec registry."""
    from microcosm.graph.codecs import SourceCodecRegistry

    snapshot_root = Path(snapshot_root)

    def load(path, *, store=None):
        del store
        owner = _source_owner()
        fraction, seed = owner.read_survey_population_request(path)
        preparation = owner.prepare_authenticated_survey_population(
            path, snapshot_root=snapshot_root, fraction=fraction, seed=seed
        )
        owner, view = _checked_preparation(preparation)
        owner.verify_survey_population_preparation(preparation)
        return view.frame

    codecs = SourceCodecRegistry()
    codecs.register(SOURCE_CODEC, load)
    return codecs


def _same_frame(expected, actual):
    """Full semantic identity; mutable-storage source seals stay in their owner."""
    _require(
        expected.schema == actual.schema
        and expected.entities == actual.entities
        and expected.links == actual.links
        and expected.metadata == actual.metadata
        and expected.mass_log == actual.mass_log
        and expected.weighted_entities == actual.weighted_entities,
        "MATERIALIZED_FRAME_CONTEXT",
    )
    try:
        for entity in expected.entities:
            pd.testing.assert_frame_equal(
                expected.table(entity),
                actual.table(entity),
                check_exact=True,
                check_flags=True,
            )
        for link in expected.links:
            pd.testing.assert_frame_equal(
                expected.link(link),
                actual.link(link),
                check_exact=True,
                check_flags=True,
            )
        pd.testing.assert_series_equal(expected.strata, actual.strata, check_exact=True)
    except (AssertionError, ValueError, TypeError):
        raise SurveyPopulationGraphError("MATERIALIZED_FRAME_VALUES") from None
    for entity in expected.weighted_entities:
        left, right = expected.weights_for(entity), actual.weights_for(entity)
        _require(
            left.kind is right.kind
            and left.values.dtype == right.values.dtype
            and left.values.tobytes() == right.values.tobytes(),
            "MATERIALIZED_FRAME_WEIGHTS",
        )


def _verify_cloned_frame(before, actual, design_weights):
    from . import puf_support

    _require(
        before.schema == actual.schema
        and before.entities == actual.entities
        and before.links == actual.links == ()
        and before.metadata == actual.metadata
        and before.mass_log == actual.mass_log
        and actual.weighted_entities == ("household",),
        "CLONE_CONTEXT",
    )
    multiplier = puf_support._id_multiplier_for_frame(before)
    try:
        # Execute the real operator's row transform one entity at a time. This
        # checks every carried cell, missing value, dtype, native key and remapped
        # membership without constructing another full cloned Frame.
        for entity in before.entities:
            expected = puf_support._clone_preassembled_entity_table(
                before.table(entity),
                entity=entity,
                schema=before.schema,
                id_multiplier=multiplier,
            )
            pd.testing.assert_frame_equal(
                expected, actual.table(entity), check_exact=True, check_flags=True
            )
        expected_strata = pd.concat([before.strata, before.strata], ignore_index=True)
        pd.testing.assert_series_equal(expected_strata, actual.strata, check_exact=True)
    except (AssertionError, ValueError, TypeError):
        raise SurveyPopulationGraphError("CLONE_FRAME_VALUES") from None
    weights = actual.weights_for("household")
    expected_weights = np.tile(before.weights_for("household").values / 2, 2)
    _require(
        weights.kind is WeightKind.IMPORTANCE
        and weights.values.tobytes() == expected_weights.tobytes(),
        "CLONE_WEIGHTS",
    )
    # The graph duplicates original design anchors along lineage. It halves
    # current IMPORTANCE weights, never the original design evidence.
    expected_design = np.tile(design_weights, 2)
    puf_support.validate_puf_clone_attachment(
        actual, boundary=PHASE, expected_fraction=1.0, expected_seed=0
    )
    return expected_design


def _check_design_anchors(population, expected):
    _require(set(population.design_weights) == {"household"}, "DESIGN_ANCHOR_SOURCES")
    actual = population.design_weights["household"]
    _require(
        type(actual) is np.ndarray
        and actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes() == expected.tobytes(),
        "DESIGN_ANCHOR_LINEAGE",
    )


def _check_population_state(population, *, version, owners, kind, ledger):
    from microcosm.graph.population import MassRecord

    _require(
        type(population.version) is str and population.version == version,
        "POPULATION_VERSION",
    )
    _require(
        dict(population.owners) == owners
        and all(type(value) is str for value in population.owners.values()),
        "POPULATION_OWNERS",
    )
    _require(
        dict(population.weight_kind) == {"household": kind}
        and type(population.weight_kind["household"]) is WeightKind,
        "POPULATION_WEIGHT_KIND",
    )
    _require(
        type(population.mass_ledger) is tuple
        and all(type(record) is MassRecord for record in population.mass_ledger)
        and population.mass_ledger == ledger,
        "POPULATION_MASS_LEDGER",
    )


def _final_artifact(manifest, store, *, node_id, name, type_, payload, capabilities):
    from microcosm.graph.artifact_edges import descriptor
    from microcosm.graph.keys import opaque_artifact_key

    node = manifest.node(node_id)
    expected = descriptor(
        producer=node_id,
        artifact=name,
        type_=type_,
        producer_key=node.key,
        capabilities=capabilities,
    )
    _require(
        node.typed_artifacts["outputs"].get(name) == expected,
        "FINAL_ARTIFACT_DESCRIPTOR",
    )
    key = opaque_artifact_key(node.key, name)
    _require(node.opaque_artifacts.get(name) == key, "FINAL_ARTIFACT_KEY")
    actual = store.load_bytes(key)
    _require(type(actual) is bytes and actual == payload, "FINAL_ARTIFACT_BYTES")


def _check_node_states(manifest, expected):
    from microcosm.graph.keys import _capabilities_projection

    _require(set(manifest.nodes) == set(expected), "MANIFEST_NODE_SET")
    for node_id, state in expected.items():
        node = manifest.node(node_id)
        _require(
            node.key == state["key"]
            and node.kernel_ref == state["ref"]
            and type(node.capabilities) is Capabilities
            and _capabilities_projection(node.capabilities) == state["capabilities"]
            and node.kernel_impl_hash == state["implementation"]
            and node.typed_artifacts == state["typed_artifacts"]
            and type(node.seed) is int
            and node.seed == state["seed"]
            and node.frame_key == state["frame_key"]
            and node.weight_key == state["weight_key"]
            and node.artifacts == state["artifacts"]
            and node.opaque_artifacts == state["opaque_artifacts"]
            and node.receipt == state["receipt"]
            and node.legacy_capabilities is False,
            "MANIFEST_NODE_STATE",
        )


@dataclass(frozen=True)
class SurveyPopulationRunValues:
    """Values retained from one completed run, without additional authority.

    Populations are the runner's detached observations, checked against the
    complete attached manifest Frames before return, not decoded substitutes.
    Downstream source-qualified boundaries must check the issued preparation
    and complete populations themselves. Freezing this container does not seal
    its contents, authenticate a copy, or admit a later graph descendant.
    """

    manifest: object
    preparation: object
    allocated_population: object
    clone_population: object | None
    compiled: object
    store: object
    kernels: object
    sources: dict


def run_authenticated_survey_population(
    source_dir,
    *,
    snapshot_root,
    store_root,
    fraction,
    seed,
    resume="auto",
    clones=True,
    return_values=False,
):
    """Reconstruct sources, then execute and verify the real selected graph.

    Warm execution can skip graph kernels; it still reconstructs both complete
    catalogues and the selected native populations. Returned manifests are
    development build records. Decoding their bytes grants no source authority.
    ``return_values=True`` returns the live composition values with the manifest;
    it does not bypass any final checks or issue a downstream admission.
    """
    from microcosm.graph import (
        ContentStore,
        Graph,
        KernelRegistry,
        SourceRef,
        compile_graph,
        run_graph,
    )
    from microcosm.graph.artifact_edges import typed_contracts
    from microcosm.graph.executor import (
        _all_node_keys,
        _expand_declared_payload,
        _expand_rewrite_coordinates,
        _input_writers,
        _source_paths_and_keys,
        _tolerance_writer_payload,
    )
    from microcosm.graph.keys import (
        _capabilities_projection,
        artifact_key,
        frame_key,
        opaque_artifact_key,
        weights_key,
    )
    from microcosm.graph.keys import seed as node_seed
    from microcosm.graph.manifest import _freeze_json
    from microcosm.graph.population import (
        _mass_record,
        expand_lineage_receipt,
        expand_writes_receipt,
        mass_record_receipt,
        weight_cap_receipt,
    )

    from .graph_sources import frame_column_declarations

    _draw_parameters(fraction, seed)
    _require(type(clones) is bool, "CLONES_FLAG")
    _require(type(return_values) is bool, "RETURN_VALUES_FLAG")
    # Do not resolve source symlinks before their authenticated owner checks them.
    source_dir, snapshot_root, store_root = (
        Path(path).absolute() for path in (source_dir, snapshot_root, store_root)
    )
    owner = _source_owner()
    preparation = owner.prepare_authenticated_survey_population(
        source_dir, snapshot_root=snapshot_root, fraction=fraction, seed=seed
    )
    owner, view = _checked_preparation(preparation)
    # Resolve '..' only after the source owner has checked the raw path and
    # refused symlinks. The executor uses this same canonical path identity.
    source_dir = source_dir.resolve(strict=True)
    payload, context_bytes, prepared_frame = view.payload, view.context, view.frame
    instructions = allocation_instructions(
        view.selection_plan, view.receipt["origins"]["households"]
    )
    columns = frame_column_declarations(prepared_frame)
    nodes = survey_population_nodes(
        columns, preparation_sha256=_sha(payload), fraction=fraction, seed=seed
    )
    _weights, allocated_context, allocation_payload, allocation_receipt, allocated = (
        _allocation_output(prepared_frame, context_bytes, instructions, _sha(payload))
    )
    design = np.array(prepared_frame.weights_for("household").values, copy=True)
    kernels = KernelRegistry()
    kernels.register(SurveyPopulationCreateKernel(preparation, source_dir=source_dir))
    kernels.register(
        SurveyPopulationAllocationKernel(
            instructions, preparation_bytes=payload, context_bytes=context_bytes
        )
    )
    clone_nodes = ()
    if clones:
        from . import graph_combined_clone as clone
        from . import puf_support
        from .graph_combined_clone import (
            register_us_combined_survey_clone_kernels,
            us_combined_survey_clone_nodes,
        )

        clone_nodes = us_combined_survey_clone_nodes(
            columns, base=ALLOCATION_NODE, source_channels=("acs", "asec")
        )
        register_us_combined_survey_clone_kernels(kernels)
    compiled = compile_graph(
        Graph("us", (SourceRef(SOURCE_NAME, SOURCE_CODEC),), (*nodes, *clone_nodes))
    )
    store = ContentStore(
        store_root, codecs=survey_population_source_codecs(snapshot_root=snapshot_root)
    )
    observed = {}
    expected_states = {}
    expected_writer_receipts = {}
    allocation_ledger = (
        _mass_record(
            prepared_frame,
            allocated,
            nodes[1],
            KernelResult(receipt=allocation_receipt),
            "declared",
        ),
    )
    all_cells = tuple(
        (entity, str(column))
        for entity in prepared_frame.entities
        for column in prepared_frame.table(entity)
    )

    def observe(node_id, population):
        owner.verify_survey_population_preparation(preparation)
        if node_id == CREATE_NODE:
            _same_frame(prepared_frame, population.frame)
            expected_design = design
            owners = dict.fromkeys(all_cells, CREATE_NODE)
            kind, ledger = WeightKind.DESIGN, ()
        elif node_id == ALLOCATION_NODE:
            _same_frame(allocated, population.frame)
            expected_design = design
            owners = dict.fromkeys(all_cells, ALLOCATION_NODE)
            kind, ledger = WeightKind.IMPORTANCE, allocation_ledger
        elif node_id in {node.id for node in clone_nodes}:
            expected_design = _verify_cloned_frame(allocated, population.frame, design)
            owners = dict.fromkeys(all_cells, clone_nodes[0].id)
            if node_id == clone_nodes[1].id:
                owners.update(
                    {(o.entity, o.column): node_id for o in clone_nodes[1].outputs}
                )
            # Full cloned cells have been checked against the real operator row
            # transform. Derive mass from these values, never the cached ledger.
            kind = WeightKind.IMPORTANCE
            ledger = (
                *allocation_ledger,
                _mass_record(
                    allocated,
                    population.frame,
                    clone_nodes[0],
                    KernelResult(),
                    "conserve",
                ),
            )
        else:
            raise SurveyPopulationGraphError("UNEXPECTED_POPULATION")
        _check_design_anchors(population, expected_design)
        state = dict(
            version=compiled.versions[node_id], owners=owners, kind=kind, ledger=ledger
        )
        _check_population_state(population, **state)
        node = compiled.graph.node(node_id)
        if node_id == CREATE_NODE:
            receipt = {
                "phase": PHASE,
                "preparation_sha256": _sha(payload),
                "source_reconstruction": "fresh_before_graph_execution",
                "release_eligible": False,
            }
        elif node_id == ALLOCATION_NODE:
            receipt = dict(allocation_receipt)
        elif node_id == clone_nodes[0].id:
            lineage, entity_facts = {}, {}
            for entity in US_SCHEMA.entities:
                lineage[entity], entity_facts[entity] = clone._entity_lineage(
                    allocated, population.frame, entity
                )
            authority = puf_support.validate_puf_clone_attachment(
                population.frame, boundary=PHASE, expected_fraction=1.0, expected_seed=0
            )
            receipt = clone.USCombinedSurveyCloneExpandKernel._receipt(
                allocated, population.frame, ("acs", "asec"), authority, entity_facts
            )
            receipt["expand"] = expand_lineage_receipt(lineage)
            receipt["expand_declared"] = _expand_declared_payload(node)
            receipt["expand_writes"] = expand_writes_receipt(
                allocated,
                population.frame,
                node,
                receipt,
                rewrite_coordinates=_expand_rewrite_coordinates(compiled, node),
            )
        else:
            receipt = {
                "phase": clone.COMBINED_CLONE_PHASE,
                "claimed_cells": sorted(
                    f"{output.entity}.{output.column}" for output in node.outputs
                ),
            }
        receipt["capabilities"] = dict(expected_nodes[node_id]["capabilities"])
        writers = _tolerance_writer_payload(
            _input_writers(compiled, node_id, receipts=expected_writer_receipts)
        )
        if writers:
            receipt["capabilities"]["tolerance_writers"] = writers
        if node.structural not in {StructuralDelta.NONE, StructuralDelta.CREATE}:
            receipt["mass"] = {
                **receipt.get("mass", {}),
                **mass_record_receipt(ledger[-1]),
            }
        receipt.update(weight_cap_receipt(population, node))
        expected_nodes[node_id]["receipt"] = _freeze_json(receipt)
        # Value-only input for the generic writer analysis. This is neither a
        # returned NodeReceipt nor a source/Frame authority object.
        expected_writer_receipts[node_id] = SimpleNamespace(
            receipt=expected_nodes[node_id]["receipt"]
        )
        _require(node_id not in observed, "DUPLICATE_POPULATION_OBSERVATION")
        observed[node_id] = population
        expected_states[node_id] = state

    # Derive expectations independently of returned/cache receipts. This is one
    # additional streaming source-key pass, not another draw or native build.
    _paths, source_keys = _source_paths_and_keys(
        compiled, {SOURCE_NAME: source_dir}, store
    )
    keys, implementations = _all_node_keys(compiled, kernels, source_keys)
    expected_nodes = {}
    for node in compiled.graph.nodes:
        key = keys[node.id]
        structural = node.structural is not StructuralDelta.NONE
        cells = (
            all_cells
            if structural
            else tuple((output.entity, output.column) for output in node.outputs)
        )
        weight_entity = (
            node.weights.entity
            if node.weights is not None
            else node.params.get("expand_weight_entity")
            if node.structural is StructuralDelta.EXPAND
            else None
        )
        expected_nodes[node.id] = {
            "key": key,
            "ref": node.kernel,
            "capabilities": _capabilities_projection(
                kernels.get(node.kernel).capabilities
            ),
            "implementation": implementations[node.id],
            "typed_artifacts": typed_contracts(compiled, node, keys, kernels),
            "seed": node_seed(key),
            "frame_key": frame_key(key) if structural else None,
            "weight_key": weights_key(key, weight_entity) if weight_entity else None,
            "artifacts": {(e, c): artifact_key(key, e, c) for e, c in cells},
            "opaque_artifacts": {
                output.name: opaque_artifact_key(key, output.name)
                for output in node.artifact_outputs
            },
        }
    manifest = run_graph(
        compiled,
        sources={SOURCE_NAME: source_dir},
        store=store,
        kernels=kernels,
        resume=resume,
        _population_observer=observe,
    )
    _require(tuple(observed) == compiled.order, "POPULATION_OBSERVER_COVERAGE")
    _check_node_states(manifest, expected_nodes)
    # Explicit store reads are necessary: the observer sees real populations,
    # while typed byte artifacts pass through a different materialization path.
    for node_id, name, type_, raw, capabilities in (
        (
            CREATE_NODE,
            "preparation",
            PREPARATION_TYPE,
            payload,
            SurveyPopulationCreateKernel.capabilities,
        ),
        (
            CREATE_NODE,
            "frame_context",
            US_FRAME_CONTEXT_TYPE,
            context_bytes,
            SurveyPopulationCreateKernel.capabilities,
        ),
        (
            ALLOCATION_NODE,
            "allocation",
            ALLOCATION_TYPE,
            allocation_payload,
            SurveyPopulationAllocationKernel.capabilities,
        ),
        (
            ALLOCATION_NODE,
            "frame_context",
            US_FRAME_CONTEXT_TYPE,
            allocated_context,
            SurveyPopulationAllocationKernel.capabilities,
        ),
    ):
        _final_artifact(
            manifest,
            store,
            node_id=node_id,
            name=name,
            type_=type_,
            payload=raw,
            capabilities=capabilities,
        )
    # Construct the optional value container before terminal checks. A caller
    # still needs the downstream issuer's independent reconstruction; this is
    # deliberately not another source certificate or a manifest decoder.
    result = (
        SurveyPopulationRunValues(
            manifest=manifest,
            preparation=preparation,
            allocated_population=observed[ALLOCATION_NODE],
            clone_population=observed[clone_nodes[-1].id] if clone_nodes else None,
            compiled=compiled,
            store=store,
            kernels=kernels,
            sources={SOURCE_NAME: source_dir},
        )
        if return_values
        else manifest
    )
    # Finish all potentially expensive owner I/O before the final pure checks
    # of graph-owned Frames. A callback during the last producer/source check
    # must not mutate an already-observed derivative and escape detection.
    _owner, final_view = _checked_preparation(preparation)
    _require(
        final_view.payload == payload
        and final_view.context == context_bytes
        and final_view.frame is prepared_frame,
        "FINAL_PREPARATION_SEAL",
    )
    _same_frame(prepared_frame, manifest.population(CREATE_NODE))
    _same_frame(allocated, manifest.population(ALLOCATION_NODE))
    for node_id in (CREATE_NODE, ALLOCATION_NODE):
        _check_design_anchors(observed[node_id], design)
    for node in clone_nodes:
        cloned_design = _verify_cloned_frame(
            allocated, manifest.population(compiled.versions[node.id]), design
        )
        _check_design_anchors(observed[node.id], cloned_design)
    for node_id, state in expected_states.items():
        # Observations are detached from executable/store populations. They can
        # be retained for composition, so seal their values after the last I/O
        # as well as the separately checked manifest Frames.
        _same_frame(
            manifest.population(compiled.versions[node_id]), observed[node_id].frame
        )
        _check_population_state(observed[node_id], **state)
        _require(
            manifest.mass_ledger(compiled.versions[node_id]) == state["ledger"],
            "FINAL_MASS_LEDGER",
        )
    _check_node_states(manifest, expected_nodes)
    return result


def _provenance_columns(entity):
    return (
        support_channel_column(entity),
        support_source_id_column(entity),
        spine_source_id_column(entity),
        support_clone_index_column(entity),
    )


def survey_population_nodes(
    columns: Sequence[Owned], *, preparation_sha256: str, fraction: Fraction, seed: int
) -> tuple[Node, Node]:
    """Declare selected CREATE then allocation, without an invented FILTER.

    The digest names checked preparation bytes; accepting it here authenticates
    nothing. The country runner must reconstruct the preparation and verify the
    materialized populations and artifacts on both cold and cached execution.
    """
    _require(type(columns) in (tuple, list), "COLUMNS")
    columns = tuple(columns)
    _require(columns and all(type(column) is Owned for column in columns), "COLUMNS")
    _require(
        all(
            column.entity in US_SCHEMA.entities
            and column.rows == "all"
            and column.ownership is Ownership.PRODUCED
            and column.rewrite is False
            for column in columns
        ),
        "CREATE_COLUMN_CONTRACT",
    )
    inventory = {(column.entity, column.column): column for column in columns}
    _require(len(inventory) == len(columns), "DUPLICATE_COLUMNS")
    for entity in US_SCHEMA.entities:
        for name in _provenance_columns(entity):
            column = inventory.get((entity, name))
            _require(column is not None, "MISSING_PROVENANCE_COLUMN")
            _require(
                column.dtype
                == ("string" if name == support_channel_column(entity) else "int64"),
                "PROVENANCE_DTYPE",
            )
    fraction_pair, seed = _draw_parameters(fraction, seed)
    digest = _digest(preparation_sha256)
    create = Node(
        id=CREATE_NODE,
        kernel=CREATE_REF,
        structural=StructuralDelta.CREATE,
        sources=(SOURCE_NAME,),
        outputs=columns,
        params={
            "phase": PHASE,
            "preparation_sha256": digest,
            "fraction": fraction_pair,
            "sample_seed": seed,
        },
        artifact_outputs=(
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ArtifactOutput("preparation", PREPARATION_TYPE),
        ),
    )
    allocate = Node(
        id=ALLOCATION_NODE,
        kernel=ALLOCATION_REF,
        base=CREATE_NODE,
        structural=StructuralDelta.REWEIGHT,
        inputs=tuple(
            Slice(entity, _provenance_columns(entity)) for entity in US_SCHEMA.entities
        ),
        weights=WeightTransition("household", "importance", mass="declared"),
        mass="declared",
        params={"phase": PHASE, "preparation_sha256": digest},
        artifact_inputs=(
            ArtifactInput(
                "frame_context", CREATE_NODE, "frame_context", US_FRAME_CONTEXT_TYPE
            ),
            ArtifactInput("preparation", CREATE_NODE, "preparation", PREPARATION_TYPE),
        ),
        artifact_outputs=(
            ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
            ArtifactOutput("allocation", ALLOCATION_TYPE),
        ),
    )
    return create, allocate
