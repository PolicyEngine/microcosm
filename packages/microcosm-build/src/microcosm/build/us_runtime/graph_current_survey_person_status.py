"""Four explicit, private graph operations for descriptive ACS/ASEC status.

The retained source owner admits literals. This fragment exposes their schema,
reuses the one descriptive recoder and binds exact original-person identities to
both initial clones. It assigns no statutory disability, blindness or annual
student eligibility. The host retains authority across execution and replay.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

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

from . import current_survey_health_coverage as health
from . import current_survey_person_status as status
from . import current_survey_person_status_source as source
from . import support_provenance as provenance
from . import survey_population_replay as replay
from .graph_survey_population import SOURCE_NAME

require = status.require
qualify_current_survey_person_status = source.qualify_current_survey_person_status
PREFIX = "survey_person_status"
SOURCE_NODE, LITERALS_NODE, RECODE_NODE, BIND_NODE = (
    PREFIX + "." + part for part in ("source_projection", "literals", "recode", "bind")
)
NODE_IDS = (SOURCE_NODE, LITERALS_NODE, RECODE_NODE, BIND_NODE)
REFS = tuple("us." + node + "@1" for node in NODE_IDS)
PROJECTION_TYPE = ArtifactType("microcosm.us.survey_person_status_projection", 1)
COLUMNS_TYPE = ArtifactType("microcosm.us.survey_person_status_columns", 1)
BINDING_TYPE = ArtifactType("microcosm.us.survey_person_status_binding", 1)
MAX_ARTIFACT_BYTES = 2**31
SOURCE_TOKENS = (
    ("person_status_native_person_id", "int64"),
    ("person_status_source_arm", "string"),
)
LITERAL_TOKENS = tuple(
    ("person_status_source_" + c, "string") for c in status.RAW_COLUMNS
)
# This is a schema, not a second recoder. It includes absent opposite-survey
# fields even when a selected source contains only one survey arm.
_CODE_FIELDS = (
    *(i.asec for i in status.ITEMS),
    "PRPERTYP",
    "PRDISFLG",
    "A_ENRLW",
    "A_FTPT",
    "A_HSCOL",
    *(i.acs for i in status.ITEMS),
    "DIS",
    "SCH",
    "SCHG",
)
_FLAG_FIELDS = (
    *(i.asec_flag for i in status.ITEMS),
    "AXENRLW",
    "AXFTPT",
    "AXHSCOL",
    *(i.acs_flag for i in status.ITEMS),
    "FDISP",
    "FSCHP",
    "FSCHGP",
)
PARSE_TOKENS = tuple(
    ("person_status_source_" + field + suffix, token)
    for field in (*_CODE_FIELDS, *_FLAG_FIELDS)
    for suffix, token in (
        ("__code", "Int64"),
        ("__literal_status", "string"),
        *((("__meaning", "string"),) if field in _FLAG_FIELDS else ()),
    )
)
BIND_TOKENS = (
    ("survey_status_source", "string"),
    ("survey_status_original_age", "Int64"),
    ("survey_status_observation_year", "Int64"),
    ("survey_status_reference_period", "string"),
    ("survey_student_reference_period", "string"),
    ("survey_student_full_time_measured", "boolean"),
    ("survey_student_annual_five_month_status_validated", "boolean"),
    ("survey_status_canonical_eligibility_assigned", "boolean"),
    *(
        (item.observation + suffix, token)
        for item in status.ITEMS
        for suffix, token in (
            ("", "boolean"),
            ("__known", "boolean"),
            ("__status", "string"),
            ("__applicable", "boolean"),
        )
    ),
    ("survey_any_applicable_difficulty", "boolean"),
    ("survey_any_applicable_difficulty__known", "boolean"),
    ("survey_any_applicable_difficulty__complete", "boolean"),
    ("survey_any_applicable_difficulty__coherent", "boolean"),
    ("survey_any_applicable_difficulty__applicable_fields", "string"),
    ("survey_publisher_disability_recode", "boolean"),
    ("survey_publisher_disability_recode__known", "boolean"),
    ("survey_publisher_disability_recode__status", "string"),
    ("survey_publisher_disability_recode__agrees_with_battery", "boolean"),
    *(
        (name + suffix, token)
        for name in status.OBSERVATIONS[-2:]
        for suffix, token in (
            ("", "boolean"),
            ("__known", "boolean"),
            ("__status", "string"),
        )
    ),
    ("survey_school_level_separate_allocation_measured", "boolean"),
)
RECODE_TOKENS = (*PARSE_TOKENS, *BIND_TOKENS)
BIND_COLUMNS = tuple(name for name, _ in BIND_TOKENS)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _modules():
    return (
        sys.modules[__name__],
        source,
        status,
        source.original,
        source.student,
        source.physical,
        source.source_csv_builtin,
        source.original.records,
        source.student.restoration,
        sys.modules[source.student.current_money_content_sha256.__module__],
        sys.modules[source.student.ReadyCurrentMoney.__module__],
        source.source,
        health,
        provenance,
        population_ops,
        replay,
    )


def _source_bytes():
    return tuple((m.__name__, _sha(Path(m.__file__).read_bytes())) for m in _modules())


def _live():
    result = {"source_owner": source._live()}
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = source.source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = (
                            source.source._function_seal(function)
                        )
    result["contract"] = source.source._runtime_marker(
        (
            PREFIX,
            NODE_IDS,
            REFS,
            tuple(asdict(t) for t in (PROJECTION_TYPE, COLUMNS_TYPE, BINDING_TYPE)),
            MAX_ARTIFACT_BYTES,
            SOURCE_NAME,
            SOURCE_TOKENS,
            LITERAL_TOKENS,
            _CODE_FIELDS,
            _FLAG_FIELDS,
            PARSE_TOKENS,
            BIND_TOKENS,
            RECODE_TOKENS,
            BIND_COLUMNS,
            asdict(_StatusKernel.capabilities),
        )
    )
    return result


def person_status_seal(qualified):
    """Pure detached-value seal; this deliberately does not grant admission."""
    require(type(qualified) is source.QualifiedSurveyPersonStatus, "QUALIFIED_TYPE")
    require(type(qualified.receipt) is bytes, "RECEIPT_TYPE")
    return (
        id(qualified),
        id(qualified.source_frame),
        id(qualified.origins),
        id(qualified.raw),
        id(qualified.observations),
        (
            source.source._function_seal(qualified._revalidate)
            if type(qualified._revalidate) is FunctionType
            else source.source._runtime_marker(qualified._revalidate)
        ),
        source.source._frame_identity(qualified.source_frame),
        source._table_seal(qualified.origins),
        source._table_seal(qualified.raw),
        source._table_seal(qualified.observations),
        qualified.receipt,
    )


def _pure_owner(qualified):
    """Recheck the retained issuer closure after callbacks, without source I/O.

    This is the pure tail of the existing owner's exact revalidation closure.
    It is not an alternate issuer: kernel construction first calls validate()
    on this exact object, and the common host requalifies after its final I/O.
    """
    callback = qualified._revalidate
    require(
        type(callback) is FunctionType and callback.__code__ is source._REVALIDATE_CODE,
        "RETAINED_OWNER_REQUIRED",
    )
    bound = dict(
        zip(
            callback.__code__.co_freevars,
            (cell.cell_contents for cell in callback.__closure__),
            strict=True,
        )
    )
    require(
        bound["result"] is qualified and bound["revalidate"] is callback,
        "RETAINED_OBJECT_CHANGED",
    )
    preparation, entry, state = bound["preparation"], bound["entry"], bound["state"]
    native, issued = bound["native"], bound["issued"]
    require(
        source.source._ISSUED.get(id(preparation)) is entry
        and entry[0]() is preparation
        and preparation.payload == entry[1]
        and entry[2] is state
        and qualified.source_frame is state.frame
        and source.source.asec_native._ISSUED.get(id(native)) is issued
        and issued[0]() is native
        and native.payload == issued[1]
        and issued[2].parent is bound["parent"]
        and source.source.acs_catalogue._lookup(state.catalogues[0])
        is bound["acs_owned"],
        "RETAINED_OWNER_CHANGED",
    )
    source.source._pure_final(state)
    controls, ready = bound["controls"], bound["ready"]
    require(
        controls._header == bound["controls_header"]
        and controls._body == bound["controls_body"]
        and ready.header == bound["ready_header"]
        and source.student.current_money_content_sha256(ready)
        == bound["ready_content"],
        "RETAINED_STUDENT_CHANGED",
    )
    require(
        qualified.origins is bound["origins"]
        and qualified.raw is bound["raw"]
        and qualified.observations is bound["observations"]
        and qualified.receipt == bound["receipt"]
        and tuple(
            source._table_seal(t)
            for t in (qualified.origins, qualified.raw, qualified.observations)
        )
        == bound["seals"],
        "RETAINED_PROJECTION_CHANGED",
    )


def _typed(records, index, tokens):
    table = pd.DataFrame(index=index.copy())
    for name, token in tokens:
        table[name] = pd.array(
            [r.get(name) for r in records], dtype=population_ops.dtype_for_token(token)
        )
    return table


def status_tables(qualified):
    """Reconstruct all outputs from raw literals and compare the owned recodes."""
    person_status_seal(qualified)
    index = qualified.origins.index
    require(
        index.dtype == np.dtype("int64")
        and index.name == "person_id"
        and index.is_unique
        and qualified.raw.index.equals(index)
        and qualified.observations.index.equals(index)
        and tuple(qualified.raw) == status.RAW_COLUMNS,
        "SOURCE_TABLE_AXES",
    )
    source_people = qualified.source_frame.person
    require(
        np.array_equal(source_people.person_id.to_numpy(), index.to_numpy()),
        "ORIGINAL_AXIS",
    )
    rows = []
    for pid, arm in qualified.origins.source.items():
        raw = qualified.raw.loc[pid].to_dict()
        row = status.recode_person_status(raw, survey=arm)
        row["survey_any_applicable_difficulty__applicable_fields"] = json.dumps(
            row["survey_any_applicable_difficulty__applicable_fields"],
            separators=(",", ":"),
        )
        require(
            set(row) <= {n for n, _ in (*LITERAL_TOKENS, *RECODE_TOKENS)},
            "UNDECLARED_RECODE",
        )
        rows.append(row)
    complete = _typed(rows, index, (*LITERAL_TOKENS, *RECODE_TOKENS))
    require(set(qualified.observations) <= set(complete), "UNDECLARED_OBSERVATION")
    for column in qualified.observations:
        require(
            qualified.observations[column].equals(complete[column]),
            "RECODE_QUALIFIER_DISAGREEMENT",
        )
    for raw, (name, _) in zip(status.RAW_COLUMNS, LITERAL_TOKENS, strict=True):
        require(
            population_ops.storage_equal(qualified.raw[raw], complete[name]),
            "RAW_QUALIFIER_DISAGREEMENT",
        )
    return (
        complete.loc[:, [n for n, _ in LITERAL_TOKENS]].copy(),
        complete.loc[:, [n for n, _ in RECODE_TOKENS]].copy(),
    )


def _literal(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return {"integer_literal": str(int(value))}
    require(type(value) is str, "ARTIFACT_LITERAL_TYPE")
    return value


def _table_document(table):
    return {
        "index": [str(int(v)) for v in table.index],
        "columns": [(c, str(table[c].dtype)) for c in table],
        "rows": [
            [_literal(v) for v in row]
            for row in table.itertuples(index=False, name=None)
        ],
    }


def projection_payload(qualified):
    """Only this private artifact carries rows; typed column artifacts carry seals."""
    payload = canonical_json(
        {
            "protocol": status.PROTOCOL,
            "source_receipt_sha256": _sha(qualified.receipt),
            "origins": _table_document(qualified.origins),
            "raw": _table_document(qualified.raw),
        }
    )
    require(len(payload) <= MAX_ARTIFACT_BYTES, "PROJECTION_BYTES")
    return payload


def _columns_payload(table):
    return canonical_json(
        {
            "protocol": status.PROTOCOL,
            "rows": len(table),
            "columns": [
                (c, population_ops.token_for_dtype(table[c].dtype)) for c in table
            ],
            "table_sha256": _sha(canonical_json(_table_document(table))),
        }
    )


def _projection_edge():
    return ArtifactInput(
        "person_status_projection", SOURCE_NODE, "projection", PROJECTION_TYPE
    )


def _columns_edge(node, alias):
    return ArtifactInput(alias, node, "columns", COLUMNS_TYPE)


def binding_edge():
    return ArtifactInput("person_status_binding", BIND_NODE, "binding", BINDING_TYPE)


def _check_pins(host_edges, host_pins):
    require(
        type(host_edges) is tuple
        and all(type(e) is ArtifactInput for e in host_edges)
        and len({e.name for e in host_edges}) == len(host_edges)
        and type(host_pins) is dict
        and set(host_pins) == {e.name for e in host_edges},
        "HOST_PINS",
    )
    for pin in host_pins.values():
        require(
            type(pin) is dict
            and set(pin) == {"producer_key", "artifact_key", "payload_sha256"}
            and all(
                type(v) is str and len(v) == 64 and not set(v) - set("0123456789abcdef")
                for v in pin.values()
            ),
            "HOST_PIN_DIGESTS",
        )


def _identities():
    return tuple(
        f("person")
        for f in (
            provenance.support_source_id_column,
            provenance.support_clone_index_column,
            provenance.spine_source_id_column,
            provenance.support_channel_column,
        )
    )


def current_survey_person_status_nodes(
    qualified, receiving_frame, *, receiving_version, after, host_edges, host_pins
):
    person_status_seal(qualified)
    require(
        type(receiving_version) is str
        and bool(receiving_version)
        and receiving_version not in NODE_IDS
        and type(after) is ArtifactInput,
        "FRAGMENT_INPUT",
    )
    _check_pins(host_edges, host_pins)
    require(after in host_edges, "ORDERING_EDGE_UNPINNED")
    require(
        not set(BIND_COLUMNS) & set(receiving_frame.person),
        "ATTACH_OWNERSHIP_COLLISION",
    )
    raw, recoded = status_tables(qualified)
    params = {
        "protocol": status.PROTOCOL,
        "source_receipt_sha256": _sha(qualified.receipt),
        "projection_sha256": _sha(projection_payload(qualified)),
        "host_edges": canonical_json(host_pins).decode(),
        "descriptive_only": True,
        "periods": canonical_json(
            [
                status.ASEC_PERIOD,
                status.ACS_PERIOD,
                status.ASEC_STUDENT_PERIOD,
                status.ACS_STUDENT_PERIOD,
            ]
        ).decode(),
        "difficulty_definitions": canonical_json(
            [asdict(item) for item in status.ITEMS]
        ).decode(),
    }

    def owned(tokens):
        return tuple(Owned("person", name, token) for name, token in tokens)

    return (
        Node(
            SOURCE_NODE,
            REFS[0],
            structural=StructuralDelta.CREATE,
            sources=(SOURCE_NAME,),
            outputs=owned(SOURCE_TOKENS),
            params=params,
            artifact_inputs=host_edges,
            artifact_outputs=(ArtifactOutput("projection", PROJECTION_TYPE),),
            description="Retain actual ACS/ASEC person-source authority and private original-support lineage.",
        ),
        Node(
            LITERALS_NODE,
            REFS[1],
            population=SOURCE_NODE,
            outputs=owned(LITERAL_TOKENS),
            params=params,
            artifact_inputs=(_projection_edge(),),
            artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
            description="Expose all literal status items, original ages and distinct allocation/edit flags; absent survey-arm items remain null.",
        ),
        Node(
            RECODE_NODE,
            REFS[2],
            population=SOURCE_NODE,
            inputs=(Slice("person", (SOURCE_TOKENS[1][0], *raw.columns)),),
            outputs=owned(RECODE_TOKENS),
            params=params,
            artifact_inputs=(
                _projection_edge(),
                _columns_edge(LITERALS_NODE, "person_status_literals"),
            ),
            artifact_outputs=(ArtifactOutput("columns", COLUMNS_TYPE),),
            description="Apply the single source recoder to observed literals; expose parse states, descriptive difficulty/student observations, unknownness, universes and reference periods.",
        ),
        Node(
            BIND_NODE,
            REFS[3],
            population=receiving_version,
            inputs=(Slice("person", _identities()),),
            outputs=owned(BIND_TOKENS),
            params=params,
            artifact_inputs=(
                *host_edges,
                _projection_edge(),
                _columns_edge(RECODE_NODE, "person_status_recodes"),
            ),
            artifact_outputs=(ArtifactOutput("binding", BINDING_TYPE),),
            description="Join descriptive status onto both initial clone families by exact original/native/arm identity; preserve all unrelated cells and weights.",
        ),
    )


def _expected_artifacts(qualified):
    raw, recoded = status_tables(qualified)
    return {
        (SOURCE_NODE, "projection"): projection_payload(qualified),
        (LITERALS_NODE, "columns"): _columns_payload(raw),
        (RECODE_NODE, "columns"): _columns_payload(recoded),
    }


def _check_artifacts(node, artifacts, qualified):
    require(set(artifacts) == {e.name for e in node.artifact_inputs}, "ARTIFACT_ROSTER")
    expected, pins = (
        _expected_artifacts(qualified),
        json.loads(node.params["host_edges"]),
    )
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
        else:
            require(
                pins.get(edge.name)
                == {
                    "producer_key": value.producer_key,
                    "artifact_key": value.key,
                    "payload_sha256": _sha(value.payload),
                },
                "HOST_EDGE_PIN",
            )


def _projection_frame(qualified):
    original, tables = qualified.source_frame, {}
    for entity in original.entities:
        columns = [original.schema.entity_id_column(entity)]
        if entity == original.schema.person_entity:
            columns += [
                original.schema.membership_column(group)
                for group in original.schema.group_entities
            ]
        tables[entity] = original.table(entity).loc[:, columns].copy()
    tables.update({name: original.link(name).copy() for name in original.links})
    tables["person"][SOURCE_TOKENS[0][0]] = qualified.origins.native_person_id.to_numpy(
        copy=True
    )
    tables["person"][SOURCE_TOKENS[1][0]] = pd.array(
        qualified.origins.source, dtype=source.STRING
    )
    return Frame(
        tables,
        original.schema,
        dict(original._weights),
        original.strata,
        metadata=original.metadata,
        mass_log=original.mass_log,
    )


def _result(qualified, node, people):
    raw, recoded = status_tables(qualified)
    if node.id == SOURCE_NODE:
        return KernelResult(
            frame=_projection_frame(qualified),
            artifacts={"projection": projection_payload(qualified)},
        )
    require(people is not None, "ORDINARY_INCOMING")
    if node.id == BIND_NODE:
        columns = health.attach_columns(
            qualified.origins,
            SimpleNamespace(person=people),
            recoded.loc[:, list(BIND_COLUMNS)],
        )
        receipt = {
            "protocol": status.PROTOCOL,
            "projection_sha256": _sha(projection_payload(qualified)),
            "source_receipt_sha256": _sha(qualified.receipt),
            "receiving_version": node.population,
            "rows": len(people),
            "source_rows": len(raw),
            "descriptive_only": True,
            "source_admission_issued": False,
            "canonical_eligibility_assigned": False,
            "release_eligible": False,
        }
        return KernelResult(
            columns=columns,
            artifacts={"binding": canonical_json(receipt)},
            receipt=receipt,
        )
    require(
        np.array_equal(people.person_id.to_numpy(), raw.index.to_numpy()),
        "SOURCE_PERSON_AXIS",
    )
    if node.id == RECODE_NODE:
        actual = people.set_index("person_id")
        require(
            population_ops.storage_equal(
                actual[SOURCE_TOKENS[1][0]],
                pd.Series(
                    pd.array(qualified.origins.source, dtype=source.STRING),
                    index=raw.index,
                    name=SOURCE_TOKENS[1][0],
                ),
            ),
            "RECODE_SOURCE_ARM",
        )
        for name in raw:
            require(
                population_ops.storage_equal(actual[name], raw[name]),
                "RECODE_SOURCE_SLICE",
            )
        table = recoded
    else:
        require(node.id == LITERALS_NODE, "NODE_ID")
        table = raw
    return KernelResult(
        columns={("person", c): table[c].copy() for c in table},
        artifacts={"columns": _columns_payload(table)},
    )


def expected_person_status_population(incoming, *, qualified, node, artifacts):
    """Pure complete reconstruction, independently of executor/store outputs."""
    _check_artifacts(node, artifacts, qualified)
    result = _result(
        qualified, node, None if incoming is None else incoming.frame.person
    )
    if node.structural is StructuralDelta.CREATE:
        require(incoming is None, "CREATE_INCOMING")
        population = population_ops.Population.from_frame(result.frame, node.id)
    else:
        require(incoming is not None, "ORDINARY_INCOMING")
        population = population_ops.patch(incoming, node, result)
    return population, result


def verify_materialized_person_status(population, incoming, *, qualified, node):
    require(node.id == BIND_NODE, "VERIFY_NODE")
    result = _result(qualified, node, incoming.frame.person)
    replay.same_replayed_population(
        population_ops.patch(incoming, node, result), population
    )
    return result.artifacts["binding"]


def _result_seal(result):
    return (
        None if result.frame is None else source.source._frame_identity(result.frame),
        tuple(
            (e, c, source._table_seal(v.rename(c).to_frame()))
            for (e, c), v in sorted(result.columns.items())
        ),
        tuple(sorted(result.artifacts.items())),
        canonical_json(dict(result.receipt)),
    )


def _context_seal(context):
    return (
        context.node,
        tuple(
            (name, source._table_seal(table))
            for name, table in sorted(context.tables.items())
        ),
        tuple(
            (name, value.type, value.key, value.producer_key, value.payload)
            for name, value in sorted(context.artifacts.items())
        ),
    )


class _StatusKernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def __init__(self, qualified, nodes, require_current, *, ref):
        require(type(require_current) is FunctionType, "HOST_CALLBACK")
        self.ref, self.qualified, self.nodes, self.require_current = (
            ref,
            qualified,
            nodes,
            require_current,
        )
        self.callback_seal = source.source._function_seal(require_current)
        self.live = _live()
        require(self.live == _LIVE, "IMPLEMENTATION_CHANGED")
        self.source_bytes = _source_bytes()
        require(self.source_bytes == _SOURCE_BYTES, "IMPLEMENTATION_SOURCE_CHANGED")
        # The original object's actual callback establishes admission once;
        # pure per-node checks do not substitute for the final host borrow.
        qualified.validate()
        self._check_live()
        _pure_owner(qualified)
        self.seal = person_status_seal(qualified)
        if ref == REFS[0]:
            self.capabilities = replace(
                self.capabilities, structural=StructuralDelta.CREATE
            )

    def _check_live(self):
        require(_live() == self.live, "IMPLEMENTATION_CHANGED")
        require(
            type(self.require_current) is FunctionType
            and source.source._function_seal(self.require_current)
            == self.callback_seal,
            "CALLBACK_CHANGED",
        )

    def implementation_hash(self):
        return source_hash(*_modules(), dependencies=self.capabilities.dependencies)

    def run(self, context):
        self._check_live()
        context_seal = _context_seal(context)
        require(_source_bytes() == self.source_bytes, "IMPLEMENTATION_SOURCE_CHANGED")
        self._check_live()
        self.require_current()
        self._check_live()
        _pure_owner(self.qualified)
        require(person_status_seal(self.qualified) == self.seal, "QUALIFIED_CHANGED")
        require(
            context.node in self.nodes and context.node.kernel == self.ref,
            "KERNEL_NODE",
        )
        _check_artifacts(context.node, context.artifacts, self.qualified)
        result = _result(self.qualified, context.node, context.tables.get("person"))
        sealed = _result_seal(result)
        self.require_current()
        self._check_live()
        require(_source_bytes() == self.source_bytes, "IMPLEMENTATION_SOURCE_CHANGED")
        self._check_live()
        _pure_owner(self.qualified)
        require(person_status_seal(self.qualified) == self.seal, "QUALIFIED_CHANGED")
        require(_result_seal(result) == sealed, "RESULT_CHANGED")
        require(_context_seal(context) == context_seal, "CONTEXT_CHANGED")
        self._check_live()
        return result


def person_status_kernels(
    qualified,
    receiving_frame,
    *,
    receiving_version,
    after,
    host_edges,
    host_pins,
    require_current,
):
    nodes = current_survey_person_status_nodes(
        qualified,
        receiving_frame,
        receiving_version=receiving_version,
        after=after,
        host_edges=host_edges,
        host_pins=host_pins,
    )
    return nodes, tuple(
        _StatusKernel(qualified, nodes, require_current, ref=ref) for ref in REFS
    )


_LIVE = _live()
_SOURCE_BYTES = _source_bytes()
