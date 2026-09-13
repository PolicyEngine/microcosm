"""Private canonical-household-role fragment: one source projection and one binding node.

This fragment is deliberately independent of the accepted country enrichment
host. It issues no run, creates no second receiving branch, changes no accepted
host declaration, cache key, source or output contract, and is appended to no
existing node roster. A host that later adopts it must extend its own reviewed
declarations explicitly and supply the real typed ancestry edges.

``survey_household_roles.source_projection`` is a CREATE node on a private branch of the
retained preparation Frame. It carries the rowwise source codes, binding and
allocation states, knownness and reasons, and emits them as a private typed
artifact. Those identities never reach a public receipt.

``survey_household_roles.bind`` owns exactly one leaf, ``is_household_head``, on the receiving
post-clone population. It fans each qualified original person onto both of its
initial clones by exact integer identity, preserves every already-known cell,
fills only null cells whose source binding is known, and refuses a conflict or
a known incumbent without a current qualified binding. When the receiving
population already carries the leaf, the node declares an explicit full-column
rewrite; otherwise it creates the column.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from microcosm.frame import Frame
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
    Slice,
    StructuralDelta,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph.canonical import canonical_json

from . import current_survey_household_roles as roles
from . import survey_population_replay as replay
from .graph_survey_population import SOURCE_NAME

require = roles.require
qualify_current_survey_household_roles = roles.qualify_current_survey_household_roles
household_role_projection_seal = roles.household_role_projection_seal
PREFIX = "survey_household_roles"
SOURCE_NODE = PREFIX + ".source_projection"
BIND_NODE = PREFIX + ".bind"
SOURCE_REF = "us." + PREFIX + ".source_projection@1"
BIND_REF = "us." + PREFIX + ".bind@1"
PROJECTION_TYPE = ArtifactType(
    "microcosm.us.current_survey_household_roles_projection", 1
)
BINDING_TYPE = ArtifactType("microcosm.us.current_survey_household_roles_binding", 1)
PROJECTION_ALIAS = "survey_household_roles_projection"
MAX_ARTIFACT_BYTES = 64 * 1024**2
_HEX = frozenset("0123456789abcdef")


def _source_bytes():
    return roles._source_bytes() + tuple(
        (module.__name__, roles._sha(Path(module.__file__).read_bytes()))
        for module in (sys.modules[__name__], population_ops, replay)
    )


def _live():
    """The existing source fence plus this fragment's executable contract."""
    result = dict(roles._live())
    for module in (sys.modules[__name__], population_ops, replay):
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = roles.source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = (
                            roles.source._function_seal(function)
                        )
    result["household_role_graph_contract"] = roles.source._runtime_marker(
        (
            PREFIX,
            SOURCE_NODE,
            BIND_NODE,
            SOURCE_REF,
            BIND_REF,
            asdict(PROJECTION_TYPE),
            asdict(BINDING_TYPE),
            PROJECTION_ALIAS,
            MAX_ARTIFACT_BYTES,
            tuple(sorted(_HEX)),
            SOURCE_NAME,
            asdict(_HouseholdRoleKernel.capabilities),
        )
    )
    return result


def country_host_edges():
    """The real typed ancestry a country host must supply, named in one place.

    The import is lazy so this fragment stays usable, and testable, without
    loading the host's model stack. Supplying these edges authenticates
    nothing by itself; the kernel still pins every incoming artifact.
    """
    from . import graph_puf_diagnostic_consumer as host

    return tuple(host.current_survey_host_edges())


def _digest(value):
    return type(value) is str and len(value) == 64 and not set(value) - _HEX


def _check_host_binding(host_edges, host_pins):
    require(
        type(host_edges) is tuple
        and all(type(edge) is ArtifactInput for edge in host_edges)
        and len({edge.name for edge in host_edges}) == len(host_edges),
        "HOST_EDGES",
    )
    require(
        type(host_pins) is dict
        and set(host_pins) == {edge.name for edge in host_edges},
        "HOST_PINS",
    )
    for pin in host_pins.values():
        require(
            type(pin) is dict
            and set(pin) == {"producer_key", "artifact_key", "payload_sha256"}
            and all(_digest(value) for value in pin.values()),
            "HOST_PIN_DIGESTS",
        )


def _projection_edge():
    return ArtifactInput(PROJECTION_ALIAS, SOURCE_NODE, "projection", PROJECTION_TYPE)


def _dtype(series):
    return population_ops.token_for_dtype(series.dtype)


def _source_outputs():
    return tuple(Owned("person", name, token) for name, token in roles.COLUMN_TOKENS)


def _identity_columns():
    return (
        roles.support_source_id_column("person"),
        roles.support_clone_index_column("person"),
        roles.spine_source_id_column("person"),
        roles.support_channel_column("person"),
    )


def _params(qualified, *, host_pins, canonical_dtype, declared_rewrite):
    document = canonical_json(
        {name: dict(pin) for name, pin in sorted(host_pins.items())}
    ).decode()
    return {
        "protocol": roles.PROTOCOL,
        "contract_sha256": roles.contract_sha256(),
        "projection_sha256": roles._sha(qualified.projection),
        "receipt_sha256": roles._sha(qualified.receipt),
        "canonical_column": roles.CANONICAL_COLUMN,
        "canonical_dtype": canonical_dtype,
        "declared_rewrite": declared_rewrite,
        "host_edges": document,
    }


def current_survey_household_roles_nodes(
    qualified, receiving_frame, *, receiving_version, host_pins, after, host_edges
):
    """Declare the two-node fragment against one explicit receiving version.

    ``receiving_frame`` decides one declaration only: whether the canonical
    leaf already exists, and therefore whether this node creates it or declares
    an explicit full-column rewrite of it. The incumbent's own declared storage
    is kept, because a rewrite must match it exactly.
    """
    household_role_projection_seal(qualified)
    require(
        type(receiving_version) is str
        and 0 < len(receiving_version) <= 256
        and receiving_version not in (SOURCE_NODE, BIND_NODE)
        and type(after) is ArtifactInput,
        "FRAGMENT_INPUT",
    )
    _check_host_binding(host_edges, host_pins)
    people = roles._person_table(receiving_frame)
    declared_rewrite = roles.CANONICAL_COLUMN in people
    canonical_dtype = roles.canonical_dtype_token(receiving_frame)
    params = _params(
        qualified,
        host_pins=host_pins,
        canonical_dtype=canonical_dtype,
        declared_rewrite=declared_rewrite,
    )
    outputs = _source_outputs()
    projection = Node(
        SOURCE_NODE,
        SOURCE_REF,
        structural=StructuralDelta.CREATE,
        sources=(SOURCE_NAME,),
        outputs=outputs,
        params=params,
        artifact_inputs=(after, *host_edges),
        artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
        description=(
            "Borrow the original ACS/ASEC person support and retain the "
            "qualified rowwise relationship codes, reference-person states, "
            "knownness and reasons on a private branch."
        ),
    )
    bind = Node(
        BIND_NODE,
        BIND_REF,
        population=receiving_version,
        inputs=(Slice("person", _identity_columns()),),
        outputs=(
            Owned(
                "person",
                roles.CANONICAL_COLUMN,
                canonical_dtype,
                rewrite=declared_rewrite,
            ),
        ),
        params=params,
        artifact_inputs=(after, _projection_edge()),
        artifact_outputs=(ArtifactOutput("binding", BINDING_TYPE),),
        description=(
            "Bind the qualified source observation to both initial clones of "
            "each original person; preserve every known cell, fill only "
            "source-known nulls, and refuse a conflict or an unbound incumbent."
        ),
    )
    return (projection, bind)


def _expected_artifacts(qualified):
    return {(SOURCE_NODE, "projection"): qualified.projection}


def _check_artifacts(node, artifacts, qualified, host_pins):
    require(set(artifacts) == {a.name for a in node.artifact_inputs}, "ARTIFACT_ROSTER")
    expected = _expected_artifacts(qualified)
    for edge in node.artifact_inputs:
        value = artifacts[edge.name]
        require(
            type(value) is ArtifactValue
            and value.type == edge.type
            and type(value.payload) is bytes
            and len(value.payload) <= MAX_ARTIFACT_BYTES,
            "ARTIFACT_TYPE",
        )
        if (edge.producer, edge.artifact) in expected:
            require(
                value.payload == expected[edge.producer, edge.artifact],
                "ARTIFACT_PAYLOAD",
            )
        elif edge.name in host_pins:
            require(
                host_pins[edge.name]
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": roles._sha(value.payload),
                },
                "HOST_EDGE_PIN",
            )
        # The ordering edge is authenticated by the owning host, which retains
        # its node and store keys. This fragment never treats its bytes as an
        # issuer, and no serialized receipt grants it source authority.


def _projection_frame(qualified):
    """A private branch that keeps original support, design and structural ids."""
    original = qualified.source_frame
    tables = {}
    for entity in original.entities:
        columns = [original.schema.entity_id_column(entity)]
        if entity == original.schema.person_entity:
            columns += [
                original.schema.membership_column(group)
                for group in original.schema.group_entities
            ]
        tables[entity] = original.table(entity).loc[:, columns].copy()
    require(
        np.array_equal(
            qualified.rows.index.to_numpy(),
            original.person.person_id.to_numpy(),
        ),
        "CREATE_ORIGIN_AXIS",
    )
    for name in roles.COLUMNS:
        tables["person"][name] = qualified.rows[name].array.copy()
    return Frame(
        tables,
        original.schema,
        dict(original._weights),
        original.strata,
        metadata=original.metadata,
        mass_log=original.mass_log,
    )


def _result(qualified, node, people):
    if node.id == SOURCE_NODE:
        return KernelResult(
            frame=_projection_frame(qualified),
            artifacts={"projection": qualified.projection},
        )
    require(node.id == BIND_NODE, "NODE_ID")
    require(people is not None, "ORDINARY_INCOMING")
    receiving = SimpleNamespace(person=people)
    require(
        roles.canonical_dtype_token(receiving) == node.params["canonical_dtype"]
        and (roles.CANONICAL_COLUMN in people) == bool(node.params["declared_rewrite"]),
        "RECEIVING_DECLARATION",
    )
    columns = roles.household_role_columns_for_population(qualified, receiving)
    receipt = roles.household_role_binding_receipt(
        qualified,
        receiving,
        receiving_version=node.population,
        declared_rewrite=bool(node.params["declared_rewrite"]),
    )
    return KernelResult(
        columns=columns,
        artifacts={"binding": canonical_json(receipt)},
        receipt=receipt,
    )


def expected_survey_household_roles_population(
    node_id, incoming, *, qualified, node, artifacts
):
    """Reconstruct the complete output independently of executor cache results."""
    require(node_id == node.id, "EXPECTED_NODE")
    _check_artifacts(node, artifacts, qualified, _decoded_pins(node))
    result = _result(
        qualified, node, None if incoming is None else incoming.frame.person
    )
    if node.structural is StructuralDelta.CREATE:
        require(incoming is None, "CREATE_INCOMING")
        return population_ops.Population.from_frame(result.frame, node.id)
    require(incoming is not None, "ORDINARY_INCOMING")
    return population_ops.patch(incoming, node, result)


def _decoded_pins(node):
    document = json.loads(node.params["host_edges"])
    require(type(document) is dict, "HOST_EDGES")
    return document


def verify_materialized_survey_household_roles(
    population, incoming, *, qualified, node
):
    """Compare the complete retained receiving population, then describe it.

    Everything this fragment does not own must survive unchanged: other
    columns, other entities, links, weights, strata, owners and the mass
    ledger. The comparison is the shared replay equality, so a drift anywhere
    in the population refuses rather than being summarized away.
    """
    require(node.id == BIND_NODE, "VERIFY_NODE")
    result = _result(qualified, node, incoming.frame.person)
    replay.same_replayed_population(
        population_ops.patch(incoming, node, result), population
    )
    receipt = dict(result.receipt)
    require(set(receipt) == roles.PUBLIC_RECEIPT_KEYS, "PUBLIC_RECEIPT_ROSTER")
    return receipt


class _HouseholdRoleKernel(KernelBase):
    """Retain the live host callback; independently qualify every executed output."""

    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, qualified, nodes, require_current, *, host_pins, ref):
        require(type(require_current) is FunctionType, "HOST_CALLBACK")
        self.ref, self.qualified, self.nodes = ref, qualified, nodes
        self.host_pins = {name: dict(pin) for name, pin in host_pins.items()}
        self.require_current = require_current
        self.callback_seal = roles.source._function_seal(require_current)
        self.live = _live()
        require(self.live == _LIVE, "IMPLEMENTATION_CHANGED")
        self.source_bytes = _source_bytes()
        require(self.source_bytes == _SOURCE_BYTES, "IMPLEMENTATION_SOURCE_CHANGED")
        self._check_live()
        self.seal = household_role_projection_seal(qualified)
        if ref == SOURCE_REF:
            self.capabilities = replace(
                self.capabilities, structural=StructuralDelta.CREATE
            )

    def _check_live(self):
        require(_live() == self.live, "IMPLEMENTATION_CHANGED")
        require(
            type(self.require_current) is FunctionType
            and roles.source._function_seal(self.require_current) == self.callback_seal,
            "CALLBACK_CHANGED",
        )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            roles,
            roles.provenance,
            population_ops,
            replay,
            roles.demographic,
            roles.demographic.load_authenticated_asec_demographic_source,
            roles.composed,
            roles.housing,
            roles.records,
            roles.source,
            roles.source.acs_native,
            roles.source.acs_native.verify_acs_native_coverage,
            roles.qualify_current_survey_household_roles,
            roles.household_role_columns_for_population,
            dependencies=self.capabilities.dependencies,
        )

    def _seal_result(self, result):
        columns = (
            None
            if not result.columns
            else tuple(
                (entity, name, str(value.dtype), roles._sha(value.to_json().encode()))
                for (entity, name), value in sorted(result.columns.items())
            )
        )
        frame = (
            None if result.frame is None else roles.source._frame_identity(result.frame)
        )
        return (
            columns,
            frame,
            tuple(sorted(result.artifacts.items())),
            canonical_json(dict(result.receipt)),
        )

    def run(self, context):
        self._check_live()
        require(_source_bytes() == self.source_bytes, "IMPLEMENTATION_SOURCE_CHANGED")
        self._check_live()
        self.require_current()
        self._check_live()
        require(
            household_role_projection_seal(self.qualified) == self.seal,
            "QUALIFIED_CHANGED",
        )
        require(
            context.node in self.nodes and context.node.kernel == self.ref,
            "KERNEL_NODE",
        )
        _check_artifacts(
            context.node, context.artifacts, self.qualified, self.host_pins
        )
        result = _result(self.qualified, context.node, context.tables.get("person"))
        sealed = self._seal_result(result)
        # Finish the owner callback and source-code identity reads before the
        # final pure checks of qualified inputs and every detached output.
        self.require_current()
        self._check_live()
        require(_source_bytes() == self.source_bytes, "IMPLEMENTATION_SOURCE_CHANGED")
        self._check_live()
        require(
            household_role_projection_seal(self.qualified) == self.seal,
            "QUALIFIED_CHANGED",
        )
        require(self._seal_result(result) == sealed, "RESULT_CHANGED")
        self._check_live()
        return result


def current_survey_household_roles_kernels(
    qualified,
    receiving_frame,
    *,
    receiving_version,
    host_pins,
    after,
    host_edges,
    require_current,
):
    """Bind an actual function callback and its code/defaults/closure identity.

    Opaque callable objects cannot supply that executable seal. Mutable owner
    objects retain their identity and remain the callback's responsibility;
    direct primitive/list/dict closure configuration must remain unchanged.
    """
    nodes = current_survey_household_roles_nodes(
        qualified,
        receiving_frame,
        receiving_version=receiving_version,
        host_pins=host_pins,
        after=after,
        host_edges=host_edges,
    )
    require(type(require_current) is FunctionType, "HOST_CALLBACK")
    return nodes, tuple(
        _HouseholdRoleKernel(
            qualified, nodes, require_current, host_pins=host_pins, ref=ref
        )
        for ref in dict.fromkeys(node.kernel for node in nodes)
    )


def register_household_role_kernels(
    registry,
    qualified,
    receiving_frame,
    *,
    receiving_version,
    host_pins,
    after,
    host_edges,
    require_current,
):
    """Register both kernels and return the declarations they answer for."""
    nodes, kernels = current_survey_household_roles_kernels(
        qualified,
        receiving_frame,
        receiving_version=receiving_version,
        host_pins=host_pins,
        after=after,
        host_edges=host_edges,
        require_current=require_current,
    )
    for kernel in kernels:
        registry.register(kernel)
    return nodes


_SOURCE_BYTES = _source_bytes()
_LIVE = _live()
