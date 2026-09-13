"""Source-owned child O/D completion on private empirical support branches.

Declarations and serialized artifacts are descriptive. The retained boundary
requires the actual preparation and an independently checked receiving parent;
the country host must call verify_materialized on cold and required replay.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit import graph_joint_empirical as adapter
from microcosm.fit import joint_empirical as empirical
from microcosm.fit import model as weight_model
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    StructuralDelta,
    canonical,
    platform_fingerprint,
    randomness,
    source_hash,
)
from microcosm.graph import population as populations
from microcosm.graph.keys import opaque_artifact_key

from . import current_child_property_income_source as child
from . import support_provenance as provenance
from . import survey_population_replay as replay
from .graph_survey_population import SOURCE_NAME

PROTOCOL = "microcosm.us.child-property-graph.v1"
PREFIX = "child_property"
DONOR = PREFIX + ".donors"
RECIPIENT = PREFIX + ".recipients"
FIT = PREFIX + ".fit"
DRAW = PREFIX + ".draw"
ATTACH = PREFIX + ".attach"
VERIFY = PREFIX + ".verify"
DONOR_REF = "us.child_property.donors@1"
RECIPIENT_REF = "us.child_property.recipients@1"
ATTACH_REF = "us.child_property.attach@1"
VERIFY_REF = "us.child_property.verify@1"
PROJECTION_TYPE = ArtifactType("microcosm.us.child_property_projection", 1)
SCENARIO_TYPE = ArtifactType("microcosm.us.child_property_scenario", 1)
COMPLETION_TYPE = ArtifactType("microcosm.us.child_property_completion", 1)
VERIFICATION_TYPE = ArtifactType("microcosm.us.child_property_verification", 1)
STATUS = "child_property_completion_status"
IMPUTED = "child_property_imputed"
WEIGHT = "original_household_design_weight"
KEY_COLUMNS = (
    "source",
    "source_year",
    "survey_year",
    "raw_native_household_id",
    "raw_native_person_id",
    "native_line_numeric_original",
)
SUFFIX = ("child-property-v1",)
MAX_ARTIFACT_BYTES = 64 * 1024**2
MAX_ORIGINALS = 600_000
require = child.require


def _json(value):
    payload = adapter._json(value)
    require(len(payload) <= MAX_ARTIFACT_BYTES, "GRAPH_ARTIFACT_SIZE")
    return payload


def _sha(value):
    return adapter._sha(value)


@dataclass(frozen=True)
class ChildPropertyOptions:
    """Explicit candidate assumption; no native support or released default."""

    scenario_id: str
    scope: str
    support: empirical.SupportRequirements
    transport: empirical.JointTransport
    stream: tuple

    def document(self):
        require(
            type(self.scenario_id) is str
            and 0 < len(self.scenario_id) <= 256
            and self.scope in ("test", "candidate")
            and type(self.support) is empirical.SupportRequirements
            and self.support.scope == self.scope
            and type(self.transport) is empirical.JointTransport,
            "SCENARIO_CONTRACT",
        )
        adapter._stream(self.stream, SUFFIX)
        return dict(
            protocol=PROTOCOL + "/scenario",
            scenario_id=self.scenario_id,
            scope=self.scope,
            support=self.support.document(),
            transport=self.transport.document(),
            stream=list(self.stream),
            coordinate_suffix=list(SUFFIX),
            donor_age_band=list(child.DONOR_AGES),
            recipient_age_band=[0, 14],
            age_transfer="explicit_same_year_asec_teenager_reference",
            source_observation_claim=False,
        )


def _modules():
    return (
        sys.modules[__name__],
        child,
        adapter,
        empirical,
        weight_model,
        randomness,
        canonical,
        provenance,
        populations,
        replay,
    )


def _source_bytes():
    modules = {m.__name__: m for m in (*child._modules(), *_modules())}
    return tuple(
        (name, _sha(Path(m.__file__).read_bytes())) for name, m in modules.items()
    )


def _live():
    result = dict(child._live())
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = child.source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = (
                            child.source._function_seal(function)
                        )
    result["child_graph_contract"] = child.source._runtime_marker(
        (
            PROTOCOL,
            PREFIX,
            DONOR,
            RECIPIENT,
            FIT,
            DRAW,
            ATTACH,
            VERIFY,
            DONOR_REF,
            RECIPIENT_REF,
            ATTACH_REF,
            VERIFY_REF,
            *(
                asdict(t)
                for t in (
                    PROJECTION_TYPE,
                    SCENARIO_TYPE,
                    COMPLETION_TYPE,
                    VERIFICATION_TYPE,
                )
            ),
            STATUS,
            IMPUTED,
            WEIGHT,
            KEY_COLUMNS,
            SUFFIX,
            MAX_ARTIFACT_BYTES,
            MAX_ORIGINALS,
            SOURCE_NAME,
            adapter.PROTOCOL,
            adapter.MAX_DRAW_BYTES,
            adapter.MAX_RECIPIENTS,
            *(
                asdict(t)
                for t in (
                    adapter.MODEL_TYPE,
                    adapter.MODEL_METADATA_TYPE,
                    adapter.DRAW_TYPE,
                )
            ),
            empirical.PROTOCOL,
            empirical.PATTERNS,
            empirical.MAX_MODEL_BYTES,
            empirical.MAX_DONORS,
            weight_model.DESIGN_WEIGHTS,
            weight_model.NO_WEIGHTS,
            weight_model.EXPLICIT_WEIGHTS,
            asdict(_ChildKernel.capabilities),
            asdict(adapter.JointEmpiricalFitKernel.capabilities),
            asdict(adapter.JointEmpiricalDrawKernel.capabilities),
        )
    )
    return result


def _origins(preparation_entry):
    document = json.loads(preparation_entry[1])["origins"]["persons"]
    result = pd.DataFrame(document["rows"], columns=document["columns"])
    for name in (
        "person_id",
        "selected_receiving_person_id",
        "selected_receiving_household_id",
        "household_id",
    ):
        require(result[name].dtype == np.dtype("int64"), "ORIGIN_INTEGER_DTYPE")
    require(result.person_id.is_unique, "ORIGINAL_PERSON_ID")
    return result.set_index("person_id")


def _key_table(keys):
    keys = tuple(keys)
    for key in keys:
        require(
            type(key) is tuple
            and len(key) == 6
            and type(key[0]) is str
            and key[0] in ("asec", "acs")
            and all(type(key[i]) is int for i in (1, 2))
            and all(type(key[i]) is str and bool(key[i]) for i in (3, 4, 5)),
            "ORIGINAL_KEY_TYPES",
        )
    require(len(set(keys)) == len(keys), "ORIGINAL_KEY_DUPLICATE")
    return pd.DataFrame(
        {
            name: pd.Series(
                [key[i] for key in keys], dtype="int64" if i in (1, 2) else "string"
            )
            for i, name in enumerate(KEY_COLUMNS)
        }
    )


def _support_frame(table):
    people = table.copy(deep=True)
    people["person_household_id"] = people.person_id.to_numpy(copy=True)
    return Frame(
        {
            "person": people,
            "household": pd.DataFrame(
                {"household_id": people.person_id.to_numpy(copy=True)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(len(people)), WeightKind.DESIGN)},
        metadata={
            "private_model_support": PROTOCOL,
            "weights": "unit_index_only; fit consumes separately authenticated explicit design column",
        },
    )


def _projections(qualified, origins):
    child.child_property_sources_seal(qualified)
    recipients = qualified.recipients
    require(
        recipients.index.identical(origins.index)
        and recipients.index.dtype == np.dtype("int64")
        and 0 < len(recipients) <= MAX_ORIGINALS,
        "ORIGINAL_RECIPIENT_ROSTER",
    )
    origin_keys = tuple(
        tuple(v) for v in origins[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    _key_table(origin_keys)
    require(
        origin_keys == tuple(recipients.recipient_key), "ORIGINAL_RECIPIENT_COORDINATES"
    )
    require(
        recipients.eligible.dtype == np.dtype("bool")
        and all(recipients.loc[recipients.source_age < 15, "eligible"])
        and not recipients.loc[
            recipients.eligible, ["ordinary_source_known", "dividend_source_known"]
        ]
        .any()
        .any()
        and recipients.loc[recipients.eligible, "reason"]
        .eq("explicit_unmeasured_child")
        .all(),
        "RECIPIENT_SOURCE_CLASS",
    )
    donor = qualified.donor_projection
    keys = tuple(donor.donors.donor_key)
    require(
        tuple(donor.donors.household_key) == tuple(key[:4] for key in keys),
        "DONOR_HOUSEHOLD_COORDINATES",
    )
    order = sorted(range(len(keys)), key=lambda i: _json(adapter._coordinate(keys[i])))
    donor_table = _key_table(tuple(keys[i] for i in order))
    donor_table.insert(0, "person_id", np.arange(1, len(order) + 1, dtype=np.int64))
    for name in (*child.TARGETS, WEIGHT):
        require(donor.donors[name].dtype == np.dtype("float64"), "DONOR_FLOAT64")
        donor_table[name] = donor.donors[name].iloc[order].to_numpy(copy=True)
    eligible = recipients.loc[recipients.eligible]
    # A nonempty original-recipient Frame remains valid when no child is
    # eligible. The generic draw declares this bool and selects internally.
    recipient_table = _key_table(tuple(recipients.recipient_key))
    recipient_table.insert(0, "person_id", recipients.index.to_numpy(copy=True))
    recipient_table["source_age"] = recipients.source_age.to_numpy(
        dtype=np.int64, copy=True
    )
    for name in ("ordinary_source_known", "dividend_source_known", "eligible"):
        require(recipients[name].dtype == np.dtype("bool"), "RECIPIENT_BOOLEAN")
        recipient_table[name] = recipients[name].to_numpy(copy=True)
    recipient_table["reason"] = pd.array(recipients.reason.tolist(), dtype="string")
    # Full-source scope excludes preparation/target sampling identity. The
    # recipient projection separately retains complete selected source context.
    donor_payload = _json(
        dict(
            protocol=PROTOCOL + "/full-donor-projection",
            table_sha256=child.physical._table_stamp(donor_table),
            full_source_seals={
                name: child._table_stamp(getattr(donor, name))
                for name in ("donors", "diagnostics", "interest", "dividend")
            },
            catalogue_sha256=qualified.evidence["catalogue_sha256"],
            weight_source="original_household_design",
            donor_rows=len(donor_table),
            donor_scope="complete_original_current_year_source",
            targets=list(child.TARGETS),
        )
    )
    recipient_payload = _json(
        dict(
            protocol=PROTOCOL + "/selected-recipient-projection",
            table_sha256=child.physical._table_stamp(recipient_table),
            original_rows_sha256=child._table_stamp(recipients),
            origins_sha256=child.physical._table_stamp(origins),
            source_evidence=qualified.evidence,
            original_rows=len(recipients),
            eligible_rows=len(eligible),
        )
    )
    return donor_table, recipient_table, donor_payload, recipient_payload


def _clone_lookup(qualified, origins, parent):
    require(type(parent) is populations.Population, "RETAINED_PARENT_TYPE")
    people, homes = parent.frame.person, parent.frame.table("household")
    pid, hid = (
        provenance.support_source_id_column("person"),
        provenance.support_source_id_column("household"),
    )
    pc, hc = (
        provenance.support_clone_index_column("person"),
        provenance.support_clone_index_column("household"),
    )
    native = provenance.spine_source_id_column("person")
    channel = provenance.support_channel_column("person")
    for table, names in (
        (people, ("person_id", "person_household_id", pid, pc, native)),
        (homes, ("household_id", hid, hc)),
    ):
        require(
            table.columns.is_unique
            and all(
                name in table and table[name].dtype == np.dtype("int64")
                for name in names
            ),
            "CLONE_INTEGER_IDENTITY",
        )
        require(table[names[0]].is_unique, "CLONE_ENTITY_DUPLICATE")
    ids = people[pid].to_numpy(copy=True)
    require(set(ids) == set(qualified.recipients.index), "CLONE_ORIGINAL_COVERAGE")
    pairs = list(zip(ids.tolist(), people[pc].tolist(), strict=True))
    require(
        len(pairs) == 2 * len(qualified.recipients)
        and len(set(pairs)) == len(pairs)
        and all(clone in (0, 1) for _, clone in pairs),
        "CLONE_PAIRS",
    )
    aligned = origins.loc[ids]
    require(
        np.array_equal(
            people[native].to_numpy(), aligned.selected_receiving_person_id.to_numpy()
        )
        and np.array_equal(
            people[channel].astype(str).to_numpy(), aligned.source.to_numpy()
        ),
        "CLONE_NATIVE_COORDINATES",
    )
    household = homes.set_index("household_id").loc[people.person_household_id]
    require(
        np.array_equal(household[hid].to_numpy(), aligned.household_id.to_numpy())
        and np.array_equal(household[hc].to_numpy(), people[pc].to_numpy()),
        "CLONE_HOUSEHOLD_MEMBERSHIP",
    )
    return ids, pd.Index(people.person_id.to_numpy(copy=True), name="person_id")


def _edge(name, producer, artifact, type_):
    return ArtifactInput(name, producer, artifact, type_)


def _bindings(host_edges, host_pins):
    require(
        type(host_edges) is tuple and all(type(e) is ArtifactInput for e in host_edges),
        "HOST_EDGES",
    )
    require(
        len({e.name for e in host_edges}) == len(host_edges)
        and set(host_pins) == {e.name for e in host_edges},
        "HOST_PIN_ROSTER",
    )
    for pin in host_pins.values():
        require(
            type(pin) is dict
            and set(pin) == {"producer_key", "artifact_key", "payload_sha256"}
            and all(adapter._hash(v) for v in pin.values()),
            "HOST_PIN",
        )


def _slices(frame, *, completion=False):
    return tuple(
        Slice(
            entity,
            tuple(frame.table(entity).columns)
            + ((STATUS, IMPUTED) if completion and entity == "person" else ()),
        )
        for entity in frame.entities
    )


def child_property_nodes(qualified, origins, parent, *, options, host_edges, host_pins):
    """Six descriptive declarations; only a retained boundary executes US nodes."""
    require(type(options) is ChildPropertyOptions, "OPTIONS_TYPE")
    _bindings(host_edges, host_pins)
    require(bool(host_edges), "PARENT_ORDERING_EDGE_REQUIRED")
    _clone_lookup(qualified, origins, parent)
    require(
        parent.version not in (DONOR, RECIPIENT, FIT, DRAW, ATTACH, VERIFY),
        "RECEIVING_VERSION",
    )
    require(
        all(
            name in parent.frame.person
            and parent.frame.person[name].dtype == np.dtype("float64")
            for name in child.TARGETS
        ),
        "RECEIVING_COMPONENT_DTYPES",
    )
    require(
        STATUS not in parent.frame.person and IMPUTED not in parent.frame.person,
        "COMPLETION_ALREADY_PRESENT",
    )
    donor, recipient, donor_payload, recipient_payload = _projections(
        qualified, origins
    )
    scenario = _json(options.document())
    params = dict(
        protocol=PROTOCOL,
        donor_sha256=_sha(donor_payload),
        recipient_sha256=_sha(recipient_payload),
        scenario_json=scenario.decode(),
        scenario_sha256=_sha(scenario),
        parent_stamp=child.physical._population_stamp(parent),
        host_pins=_json(host_pins).decode(),
    )
    declarations = []
    for name, ref, table in (
        (DONOR, DONOR_REF, donor),
        (RECIPIENT, RECIPIENT_REF, recipient),
    ):
        declarations.append(
            Node(
                name,
                ref,
                structural=StructuralDelta.CREATE,
                sources=(SOURCE_NAME,),
                outputs=tuple(
                    Owned("person", col, populations.token_for_dtype(table[col].dtype))
                    for col in table
                    if col != "person_id"
                ),
                params=params,
                artifact_inputs=host_edges,
                artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),)
                + (
                    (ArtifactOutput("scenario", SCENARIO_TYPE),)
                    if name == RECIPIENT
                    else ()
                ),
                description="Private source-owned full teenager donors with explicit original design weights."
                if name == DONOR
                else "Private original recipients and explicit child eligibility, preserving source coordinates and nonmeasurement; no clone IDs seed draws.",
            )
        )
    declarations.append(
        adapter.joint_empirical_fit_node(
            FIT,
            population=DONOR,
            entity="person",
            targets=child.TARGETS,
            donor_key_columns=KEY_COLUMNS,
            household_key_columns=KEY_COLUMNS[:4],
            weight_column=WEIGHT,
            support=options.support,
            source_projection=_edge(
                "source_projection", DONOR, "projection", PROJECTION_TYPE
            ),
            source_sha256=_sha(donor_payload),
        )
    )
    declarations.append(
        adapter.joint_empirical_draw_node(
            DRAW,
            population=RECIPIENT,
            entity="person",
            recipient_key_columns=KEY_COLUMNS,
            coordinate_suffix=SUFFIX,
            stream=options.stream,
            support=options.support,
            transport=options.transport,
            scenario_sha256=_sha(scenario),
            donor_source_sha256=_sha(donor_payload),
            source_sha256=_sha(recipient_payload),
            source_projection=_edge(
                "source_projection", RECIPIENT, "projection", PROJECTION_TYPE
            ),
            model_producer=FIT,
            eligibility_column="eligible",
        )
    )
    artifacts = (
        _edge("donor_projection", DONOR, "projection", PROJECTION_TYPE),
        _edge("recipient_projection", RECIPIENT, "projection", PROJECTION_TYPE),
        _edge("scenario", RECIPIENT, "scenario", SCENARIO_TYPE),
        _edge("model", FIT, "model", adapter.MODEL_TYPE),
        _edge("model_metadata", FIT, "model_metadata", adapter.MODEL_METADATA_TYPE),
        _edge("draws", DRAW, "draws", adapter.DRAW_TYPE),
    )
    declarations.append(
        Node(
            ATTACH,
            ATTACH_REF,
            population=parent.version,
            inputs=_slices(parent.frame),
            outputs=tuple(
                Owned("person", name, "float64", rewrite=True) for name in child.TARGETS
            )
            + (Owned("person", STATUS, "string"), Owned("person", IMPUTED, "bool")),
            params=params,
            artifact_inputs=(*host_edges, *artifacts),
            artifact_outputs=(ArtifactOutput("completion", COMPLETION_TYPE),),
            description="Reconstruct the actual paired empirical law from retained sources; fill only jointly unknown eligible children identically across the two original clones.",
        )
    )
    declarations.append(
        Node(
            VERIFY,
            VERIFY_REF,
            population=parent.version,
            inputs=_slices(parent.frame, completion=True),
            params=params,
            artifact_inputs=(
                *host_edges,
                *artifacts,
                _edge("completion", ATTACH, "completion", COMPLETION_TYPE),
            ),
            artifact_outputs=(ArtifactOutput("verification", VERIFICATION_TYPE),),
            description="Requalify source owners and compare every receiving table, effective weight and stratum to independent completion; host additionally verifies full population owners and ledger on every cold/required run.",
        )
    )
    return tuple(declarations)


def _reconstruct(qualified, origins, parent, options, nodes):
    """Use real numeric and keyed operators independently of cached artifacts."""
    donor, recipient, donor_payload, recipient_payload = _projections(
        qualified, origins
    )
    recipient_support = recipient
    recipient = recipient.loc[recipient.eligible]
    model = empirical.fit_joint_empirical(
        donor,
        targets=child.TARGETS,
        donor_keys=tuple(donor[list(KEY_COLUMNS)].itertuples(index=False, name=None)),
        household_keys=tuple(
            donor[list(KEY_COLUMNS[:4])].itertuples(index=False, name=None)
        ),
        support=options.support,
        weights=WEIGHT,
    )
    keys = tuple(recipient[list(KEY_COLUMNS)].itertuples(index=False, name=None))
    uniforms = [
        randomness.keyed_uniform(
            stream=options.stream, keys=tuple((*key, *SUFFIX, purpose) for key in keys)
        )
        for purpose in ("pattern", "donor")
    ]
    draw = empirical.draw_joint_empirical(
        model,
        pattern_uniforms=uniforms[0],
        donor_uniforms=uniforms[1],
        transport=options.transport,
    )
    draw_node = next(node for node in nodes if node.id == DRAW)
    draw_payload = _json(
        adapter._draw_document(draw, recipient.person_id, keys, draw_node.params)
    )
    metadata = _json(
        adapter._metadata(
            model,
            source_sha256=_sha(donor_payload),
            support_json=draw_node.params["support_json"],
        )
    )
    ids, index = _clone_lookup(qualified, origins, parent)
    lookup = {
        int(identity): position for position, identity in enumerate(recipient.person_id)
    }
    eligible = np.array([int(identity) in lookup for identity in ids], dtype=bool)
    columns = {}
    for axis, name in enumerate(child.TARGETS):
        values = parent.frame.person[name].to_numpy(copy=True)
        require(np.isnan(values[eligible]).all(), "CHILD_KNOWN_INCUMBENT")
        for row in np.flatnonzero(eligible):
            values[row] = draw.values[lookup[int(ids[row])], axis]
        columns["person", name] = pd.Series(
            values, index=index, name=name, dtype="float64"
        )
    status = ["not_recipient"] * len(ids)
    for row in np.flatnonzero(eligible):
        status[row] = (
            "imputed_positive"
            if draw.patterns[lookup[int(ids[row])]]
            else "imputed_zero"
        )
    columns["person", STATUS] = pd.Series(
        pd.array(status, dtype="string"), index=index, name=STATUS
    )
    columns["person", IMPUTED] = pd.Series(
        eligible, index=index, name=IMPUTED, dtype="bool"
    )
    completion = _json(
        dict(
            protocol=PROTOCOL + "/completion",
            model_sha256=model.sha256,
            support_sha256=draw_node.params["support_sha256"],
            scenario_sha256=draw_node.params["scenario_sha256"],
            donor_source_sha256=_sha(donor_payload),
            recipient_source_sha256=_sha(recipient_payload),
            draw_sha256=_sha(draw_payload),
            original_draws=json.loads(draw_payload)["rows"],
            source_knownness_changed=False,
            model_zero_origin="imputed_zero",
            pair_binding="exact_two_initial_clones",
            original_recipient_count=len(recipient),
            receiving_imputed_count=int(eligible.sum()),
        )
    )
    verification = _json(
        dict(
            protocol=PROTOCOL + "/verification",
            completion_sha256=_sha(completion),
            model_sha256=model.sha256,
            scenario_sha256=draw_node.params["scenario_sha256"],
            original_recipient_count=len(recipient),
            receiving_imputed_count=int(eligible.sum()),
            source_knownness_changed=False,
            full_population_owner_check_required=True,
        )
    )
    payloads = {
        (DONOR, "projection"): donor_payload,
        (RECIPIENT, "projection"): recipient_payload,
        (RECIPIENT, "scenario"): _json(options.document()),
        (FIT, "model"): model.to_bytes(),
        (FIT, "model_metadata"): metadata,
        (DRAW, "draws"): draw_payload,
        (ATTACH, "completion"): completion,
        (VERIFY, "verification"): verification,
    }
    result = KernelResult(
        columns=columns,
        artifacts={"completion": completion},
        receipt={
            "original_recipient_count": len(recipient),
            "receiving_imputed_count": int(eligible.sum()),
            "model_sha256": model.sha256,
            "source_observation_claim": False,
        },
    )
    attach = next(node for node in nodes if node.id == ATTACH)
    return (
        populations.patch(parent, attach, result),
        result,
        payloads,
        donor,
        recipient_support,
    )


def _artifacts(node, values, expected, host_pins):
    require(
        set(values) == {edge.name for edge in node.artifact_inputs}, "ARTIFACT_ROSTER"
    )
    producers = {}
    for edge in node.artifact_inputs:
        value = values[edge.name]
        require(
            type(value) is ArtifactValue
            and value.type == edge.type
            and value.key == opaque_artifact_key(value.producer_key, edge.artifact),
            "ARTIFACT_TYPE_OR_KEY",
        )
        if edge.name in host_pins:
            require(
                dict(
                    producer_key=value.producer_key,
                    artifact_key=value.key,
                    payload_sha256=_sha(value.payload),
                )
                == host_pins[edge.name],
                "HOST_ARTIFACT_CHANGED",
            )
        else:
            require(
                value.payload == expected[edge.producer, edge.artifact],
                "RECONSTRUCTED_ARTIFACT_CHANGED",
            )
            require(
                value.numerics.numeric is Numeric.PLATFORM_BITWISE
                and value.numerics.platform == platform_fingerprint(),
                "ARTIFACT_NUMERICS",
            )
        prior = producers.setdefault(edge.producer, value.producer_key)
        require(prior == value.producer_key, "PAIRED_ARTIFACT_PRODUCER")


def _context(context, expected):
    """Check the full declared table projection and all effective weights."""
    require(
        set(context.tables) == set(expected.frame.entities) and not context.sources,
        "CONTEXT_ENTITIES",
    )
    for entity in expected.frame.entities:
        table = expected.frame.table(entity)
        actual = context.tables[entity]
        require(set(actual) == set(table), "CONTEXT_COLUMNS")
        replay._axis(table.index, actual.index)
        for name in table:
            replay._series(table[name], actual[name])
        try:
            weights = expected.frame.resolve_weights(entity)
        except ValueError:
            require(
                entity not in context.weights
                and entity not in expected.frame.weighted_entities,
                "CONTEXT_WEIGHTS",
            )
        else:
            require(
                entity in context.weights
                and context.weights[entity].kind is weights.kind
                and replay._array_bytes_equal(
                    weights.values, context.weights[entity].values
                ),
                "CONTEXT_WEIGHTS",
            )
    replay._axis(expected.frame.strata.index, context.strata.index)
    replay._series(expected.frame.strata, context.strata)


class ChildPropertyBoundary:
    """Actual preparation custody plus a sealed country-parent callback.

    require_parent must return the exact retained Population after verifying
    its actual owner. A detached checked-view object does not issue authority.
    This fragment never issues a tax-complete or release-eligible handle.
    """

    __slots__ = (
        "preparation",
        "parent",
        "require_parent",
        "options",
        "options_bytes",
        "host_edges",
        "host_pins",
        "host_bytes",
        "callback_seal",
        "parent_stamp",
        "revoked",
        "entry",
        "origins",
        "origins_stamp",
        "qualified",
        "qualified_seal",
        "nodes",
        "declaration",
    )

    def __init__(
        self, preparation, parent, *, require_parent, options, host_edges, host_pins
    ):
        require(
            type(preparation) is child.source.AuthenticatedSurveyPopulationPreparation
            and type(parent) is populations.Population
            and type(require_parent) is FunctionType,
            "ACTUAL_OWNERS_REQUIRED",
        )
        require(
            _live() == _LIVE and _source_bytes() == _SOURCE_BYTES and _live() == _LIVE,
            "IMPLEMENTATION_CHANGED",
        )
        self.preparation, self.parent, self.require_parent = (
            preparation,
            parent,
            require_parent,
        )
        self.options = options
        self.options_bytes = _json(options.document())
        self.host_edges = tuple(host_edges)
        self.host_pins = json.loads(_json(host_pins))
        self.host_bytes = _json(self.host_pins)
        self.callback_seal = child.source._function_seal(require_parent)
        self.parent_stamp = child.physical._population_stamp(parent)
        self.revoked = False
        self.entry = preparation._checked()
        self.origins = _origins(self.entry)
        self.origins_stamp = child.physical._table_stamp(self.origins)
        self.qualified = child.qualify_child_property_sources(preparation)
        self.qualified_seal = child.child_property_sources_seal(self.qualified)
        self.nodes = child_property_nodes(
            self.qualified,
            self.origins,
            parent,
            options=options,
            host_edges=self.host_edges,
            host_pins=self.host_pins,
        )
        self.declaration = tuple(node.normative() for node in self.nodes)
        self._current()

    def _pure(self):
        require(not self.revoked and _live() == _LIVE, "BOUNDARY_REVOKED_OR_CHANGED")
        require(
            type(self.require_parent) is FunctionType
            and child.source._function_seal(self.require_parent) == self.callback_seal,
            "PARENT_CALLBACK_CHANGED",
        )
        require(
            _json(self.options.document()) == self.options_bytes
            and _json(self.host_pins) == self.host_bytes
            and tuple(node.normative() for node in self.nodes) == self.declaration,
            "BOUNDARY_CONFIGURATION_CHANGED",
        )
        require(
            child.physical._population_stamp(self.parent) == self.parent_stamp
            and child.physical._table_stamp(self.origins) == self.origins_stamp
            and child.child_property_sources_seal(self.qualified)
            == self.qualified_seal,
            "RETAINED_VALUES_CHANGED",
        )
        require(
            child.source._ISSUED.get(id(self.preparation)) is self.entry
            and self.preparation.payload == self.entry[1],
            "PREPARATION_OWNER_CHANGED",
        )
        child.source._pure_final(self.entry[2])

    def _current(self):
        self._pure()
        require(_source_bytes() == _SOURCE_BYTES, "IMPLEMENTATION_SOURCE_CHANGED")
        self._pure()
        require(self.require_parent() is self.parent, "PARENT_OWNER_IDENTITY")
        self._pure()
        require(self.preparation._checked() is self.entry, "PREPARATION_OWNER_IDENTITY")
        self._pure()
        current = child.qualify_child_property_sources(self.preparation)
        self._pure()
        require(
            child.child_property_sources_seal(current) == self.qualified_seal,
            "REQUALIFIED_VALUES_CHANGED",
        )
        require(_source_bytes() == _SOURCE_BYTES, "IMPLEMENTATION_SOURCE_CHANGED")
        self._pure()
        return current

    def kernels(self):
        self._pure()
        return tuple(
            _ChildKernel(self, ref)
            for ref in (DONOR_REF, RECIPIENT_REF, ATTACH_REF, VERIFY_REF)
        ) + (adapter.JointEmpiricalFitKernel(), adapter.JointEmpiricalDrawKernel())

    def verify_materialized(
        self, population, artifacts, *, support_populations, verification
    ):
        """Mandatory host check even when every node was a required cache hit."""
        try:
            qualified = self._current()
            expected, _, payloads, donor, recipient = _reconstruct(
                qualified, self.origins, self.parent, self.options, self.nodes
            )
            verify_node = self.nodes[-1]
            _artifacts(verify_node, artifacts, payloads, self.host_pins)
            require(
                type(verification) is ArtifactValue
                and verification.type == VERIFICATION_TYPE
                and verification.key
                == opaque_artifact_key(verification.producer_key, "verification")
                and verification.payload == payloads[VERIFY, "verification"]
                and verification.numerics.numeric is Numeric.PLATFORM_BITWISE
                and verification.numerics.platform == platform_fingerprint(),
                "MATERIALIZED_VERIFICATION",
            )
            require(
                set(support_populations) == {DONOR, RECIPIENT},
                "SUPPORT_POPULATION_ROSTER",
            )
            expected_support = {
                name: populations.Population.from_frame(_support_frame(table), name)
                for name, table in ((DONOR, donor), (RECIPIENT, recipient))
            }
            for name, support in expected_support.items():
                replay.same_replayed_population(support, support_populations[name])
            replay.same_replayed_population(expected, population)
            stamp = child.physical._population_stamp(population)
            support_stamps = {
                name: child.physical._population_stamp(value)
                for name, value in support_populations.items()
            }
            self._current()
            require(
                child.physical._population_stamp(population) == stamp,
                "FINAL_OUTPUT_CHANGED",
            )
            replay.same_replayed_population(expected, population)
            for name, support in expected_support.items():
                require(
                    child.physical._population_stamp(support_populations[name])
                    == support_stamps[name],
                    "FINAL_SUPPORT_CHANGED",
                )
                replay.same_replayed_population(support, support_populations[name])
            self._pure()
            return json.loads(payloads[VERIFY, "verification"])
        except Exception:
            self.revoked = True
            raise


class _ChildKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, boundary, ref):
        self.boundary, self.ref = boundary, ref
        if ref in (DONOR_REF, RECIPIENT_REF):
            self.capabilities = replace(
                self.capabilities, structural=StructuralDelta.CREATE
            )
        else:
            # Independent verification recomputes the declared keyed law.
            self.capabilities = replace(
                self.capabilities,
                determinism=Determinism.SEEDED,
                seed_source=SeedSource.KEYED,
            )

    def implementation_hash(self):
        return source_hash(
            *_modules(), *child._modules(), dependencies=self.capabilities.dependencies
        )

    def run(self, context):
        boundary = self.boundary
        try:
            qualified = boundary._current()
            require(
                context.node in boundary.nodes
                and context.node.kernel == self.ref
                and dict(context.params) == dict(context.node.params),
                "KERNEL_DECLARATION",
            )
            if self.ref in (DONOR_REF, RECIPIENT_REF):
                donor, recipient, donor_payload, recipient_payload = _projections(
                    qualified, boundary.origins
                )
                require(
                    not context.tables
                    and not context.weights
                    and set(context.sources) == {SOURCE_NAME}
                    and Path(context.sources[SOURCE_NAME]) == boundary.entry[2].root,
                    "SUPPORT_CREATE_CONTEXT",
                )
                _artifacts(context.node, context.artifacts, {}, boundary.host_pins)
                is_donor = self.ref == DONOR_REF
                frame = _support_frame(donor if is_donor else recipient)
                result = KernelResult(
                    frame=frame,
                    artifacts={
                        "projection": donor_payload if is_donor else recipient_payload,
                        **({} if is_donor else {"scenario": boundary.options_bytes}),
                    },
                    receipt={
                        "private_support_rows": len(frame.person),
                        "source_admission_issued": False,
                    },
                )
                result_stamp = child.source._frame_identity(frame)
            else:
                expected, result, payloads, _, _ = _reconstruct(
                    qualified,
                    boundary.origins,
                    boundary.parent,
                    boundary.options,
                    boundary.nodes,
                )
                _artifacts(
                    context.node, context.artifacts, payloads, boundary.host_pins
                )
                _context(
                    context, boundary.parent if self.ref == ATTACH_REF else expected
                )
                if self.ref == VERIFY_REF:
                    result = KernelResult(
                        artifacts={"verification": payloads[VERIFY, "verification"]},
                        receipt={
                            "original_recipient_count": int(
                                qualified.recipients.eligible.sum()
                            ),
                            "source_observation_claim": False,
                            "full_population_owner_check_required": True,
                        },
                    )
                result_stamp = tuple(
                    (key, child.physical._table_stamp(value.to_frame()))
                    for key, value in sorted(result.columns.items())
                )
            artifacts_stamp, receipt_stamp = (
                tuple(sorted(result.artifacts.items())),
                _json(dict(result.receipt)),
            )
            boundary._current()
            final_stamp = (
                child.source._frame_identity(result.frame)
                if result.frame is not None
                else tuple(
                    (key, child.physical._table_stamp(value.to_frame()))
                    for key, value in sorted(result.columns.items())
                )
            )
            require(
                result_stamp == final_stamp
                and tuple(sorted(result.artifacts.items())) == artifacts_stamp
                and _json(dict(result.receipt)) == receipt_stamp,
                "FINAL_KERNEL_RESULT_CHANGED",
            )
            boundary._pure()
            return result
        except Exception:
            boundary.revoked = True
            raise


_SOURCE_BYTES = _source_bytes()
_LIVE = _live()
