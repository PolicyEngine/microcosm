"""Typed paired empirical fit/draw operations over private model-support tables.

The host owns source qualification, support policy, eligible recipients and
attachment to its receiving population. Side-table IDs are only join identities;
only separately declared original coordinates enter the keyed random stream.
Model/draw artifacts contain private exact identifiers, never source authority.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from microcosm.fit import joint_empirical as empirical
from microcosm.fit import model as weight_model
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
    SeedSource,
    Slice,
    canonical,
    platform_fingerprint,
    randomness,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

PROTOCOL = "microcosm.fit.graph-joint-empirical.v1"
MODEL_TYPE = ArtifactType("microcosm.fit.joint_empirical_model", 1)
MODEL_METADATA_TYPE = ArtifactType("microcosm.fit.joint_empirical_metadata", 1)
DRAW_TYPE = ArtifactType("microcosm.fit.joint_empirical_draw", 1)
# The draw document carries one row per recipient. Its producer in the US
# runtime materialises it once under a whole-roster ceiling of 64 accumulations
# of 64 MiB (graph_child_property_income.MAX_ROSTER_BYTES); this decoder's
# ceiling on the same bytes is that number.
MAX_DRAW_BYTES = 64 * 64 * 1024**2
MAX_RECIPIENTS = 1_048_576


def _require(condition, reason):
    if not condition:
        raise ValueError("GRAPH_JOINT_EMPIRICAL_" + reason)


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _hash(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _document(payload, limit=MAX_DRAW_BYTES):
    _require(type(payload) is bytes and 0 < len(payload) <= limit, "PAYLOAD_BYTES")
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
        _require(type(value) is dict and _json(value) == payload, "CANONICAL_JSON")
    except (UnicodeError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError("GRAPH_JOINT_EMPIRICAL_JSON") from error
    return value


def _names(values):
    _require(
        type(values) is tuple
        and bool(values)
        and all(
            type(v) is str and v and v.strip() == v and "." not in v for v in values
        )
        and len(set(values)) == len(values),
        "COLUMN_NAMES",
    )
    return values


def _support(value):
    _require(type(value) is str, "SUPPORT_JSON")
    doc = _document(value.encode())
    _require(
        set(doc)
        == {
            "policy_id",
            "scope",
            "overall",
            "positive",
            "pattern",
            "required_patterns",
        },
        "SUPPORT_SCHEMA",
    )
    for name in ("overall", "positive", "pattern"):
        _require(
            type(doc[name]) is dict
            and set(doc[name]) == {"rows", "households", "person_ess", "household_ess"},
            "SUPPORT_SCHEMA",
        )
    _require(type(doc["required_patterns"]) is list, "SUPPORT_SCHEMA")
    result = empirical.SupportRequirements(
        doc["policy_id"],
        doc["scope"],
        *(
            empirical.SupportThreshold(**doc[name])
            for name in ("overall", "positive", "pattern")
        ),
        tuple(doc["required_patterns"]),
    )
    _require(_json(result.document()).decode() == value, "SUPPORT_CANONICAL")
    return result


def _transport(value):
    _require(type(value) is str, "TRANSPORT_JSON")
    doc = _document(value.encode())
    _require(
        set(doc) == {"receipt_factor", "amount_factors"}
        and type(doc["amount_factors"]) is list,
        "TRANSPORT_SCHEMA",
    )
    result = empirical.JointTransport(
        doc["receipt_factor"], tuple(doc["amount_factors"])
    )
    _require(_json(result.document()).decode() == value, "TRANSPORT_CANONICAL")
    return result


def _coordinate(values):
    """Reuse the numerical model's exact tagged int/string key codec."""
    return empirical._key(values)


def _stream(stream, suffix):
    _require(type(suffix) is tuple and bool(suffix), "COORDINATE_SUFFIX")
    _coordinate(suffix)
    # The maintained random protocol validates the stream, including bool seeds.
    # An empty coordinate batch validates without consuming sequential RNG state.
    randomness.keyed_uniform(stream=stream, keys=())


def _edge(edge):
    _require(
        type(edge) is ArtifactInput and edge.name == "source_projection", "SOURCE_EDGE"
    )
    return edge


def joint_empirical_fit_node(
    node_id,
    *,
    population,
    entity,
    targets,
    donor_key_columns,
    household_key_columns,
    weight_column,
    support,
    source_projection,
    source_sha256,
):
    """Fit exact paired donors; explicit weights carry no typed-design claim."""
    _names(targets)
    _names(donor_key_columns)
    _names(household_key_columns)
    _names((entity, weight_column))
    _require(
        len(targets) == 2 and type(support) is empirical.SupportRequirements,
        "FIT_PARAMETERS",
    )
    _require(_hash(source_sha256), "SOURCE_HASH")
    keys = tuple(dict.fromkeys((*donor_key_columns, *household_key_columns)))
    _require(
        not set(keys) & set((*targets, weight_column)) and weight_column not in targets,
        "COLUMN_COLLISION",
    )
    columns = (*targets, *keys, weight_column)
    _require(f"{entity}_id" not in columns, "SUPPORT_ID_IS_NOT_SOURCE_KEY")
    support_json = _json(support.document()).decode()
    return Node(
        node_id,
        JointEmpiricalFitKernel.ref,
        population=population,
        inputs=(Slice(entity, columns),),
        params=dict(
            entity=entity,
            targets=targets,
            donor_key_columns=donor_key_columns,
            household_key_columns=household_key_columns,
            weight_column=weight_column,
            weight_kind="explicit",
            support_json=support_json,
            support_sha256=_sha(support_json.encode()),
            source_sha256=source_sha256,
        ),
        artifact_inputs=(_edge(source_projection),),
        artifact_outputs=(
            ArtifactOutput("model", MODEL_TYPE),
            ArtifactOutput("model_metadata", MODEL_METADATA_TYPE),
        ),
        description="Fit private paired empirical donors with explicit weights and support diagnostics; no source or eligibility authority.",
    )


def joint_empirical_draw_node(
    node_id,
    *,
    population,
    entity,
    recipient_key_columns,
    coordinate_suffix,
    stream,
    support,
    transport,
    scenario_sha256,
    donor_source_sha256,
    source_sha256,
    source_projection,
    model_producer,
    eligibility_column=None,
):
    """Draw once per original coordinate; receiving/clone IDs never seed draws."""
    _names((entity,))
    _names(recipient_key_columns)
    if eligibility_column is not None:
        _names((eligibility_column,))
        _require(
            eligibility_column not in (*recipient_key_columns, f"{entity}_id"),
            "ELIGIBILITY_COLLISION",
        )
    _require(
        f"{entity}_id" not in recipient_key_columns, "SUPPORT_ID_IS_NOT_SOURCE_KEY"
    )
    _require(
        type(support) is empirical.SupportRequirements
        and type(transport) is empirical.JointTransport,
        "DRAW_PARAMETERS",
    )
    _require(
        all(_hash(v) for v in (scenario_sha256, donor_source_sha256, source_sha256)),
        "DRAW_HASHES",
    )
    _stream(stream, coordinate_suffix)
    support_json, transport_json = (
        _json(support.document()).decode(),
        _json(transport.document()).decode(),
    )
    return Node(
        node_id,
        JointEmpiricalDrawKernel.ref,
        population=population,
        inputs=(
            Slice(
                entity,
                recipient_key_columns
                + (() if eligibility_column is None else (eligibility_column,)),
            ),
        ),
        params=dict(
            entity=entity,
            recipient_key_columns=recipient_key_columns,
            eligibility_column=eligibility_column,
            coordinate_suffix=coordinate_suffix,
            stream=stream,
            support_json=support_json,
            support_sha256=_sha(support_json.encode()),
            transport_json=transport_json,
            transport_sha256=_sha(transport_json.encode()),
            scenario_sha256=scenario_sha256,
            donor_source_sha256=donor_source_sha256,
            source_sha256=source_sha256,
        ),
        artifact_inputs=(
            ArtifactInput("model", model_producer, "model", MODEL_TYPE),
            ArtifactInput(
                "model_metadata", model_producer, "model_metadata", MODEL_METADATA_TYPE
            ),
            _edge(source_projection),
        ),
        artifact_outputs=(ArtifactOutput("draws", DRAW_TYPE),),
        description="Draw private paired values from original-coordinate keyed streams; preserve imputation and donor lineage without changing a survey population.",
    )


def _artifact(context, name, *, platform=False):
    edges = [edge for edge in context.node.artifact_inputs if edge.name == name]
    _require(len(edges) == 1 and name in context.artifacts, "ARTIFACT_ROSTER")
    edge, value = edges[0], context.artifacts[name]
    _require(type(value) is ArtifactValue and value.type == edge.type, "ARTIFACT_TYPE")
    _require(
        value.key == opaque_artifact_key(value.producer_key, edge.artifact),
        "ARTIFACT_KEY",
    )
    if platform:
        _require(
            value.numerics.numeric is Numeric.PLATFORM_BITWISE
            and value.numerics.platform == platform_fingerprint(),
            "ARTIFACT_NUMERIC_SCOPE",
        )
    return value


def _table(context):
    _require(
        set(context.tables) == {context.params["entity"]} and not context.sources,
        "CONTEXT_TABLES",
    )
    _require(
        set(context.artifacts) == {edge.name for edge in context.node.artifact_inputs},
        "CONTEXT_ARTIFACTS",
    )
    table = context.tables[context.params["entity"]]
    identity = context.params["entity"] + "_id"
    _require(
        table.columns.is_unique
        and identity in table
        and table[identity].dtype == np.dtype("int64")
        and table[identity].is_unique,
        "SUPPORT_IDENTITY",
    )
    _require(set(context.node.inputs[0].columns) <= set(table), "TABLE_COLUMNS")
    _require(
        _sha(_artifact(context, "source_projection").payload)
        == context.params["source_sha256"],
        "SOURCE_PROJECTION_HASH",
    )
    return table


def _keys(table, columns, *, unique):
    # Column-wise indexing preserves exact scalar types; iterrows can coerce an
    # integer ID to float when the same row also contains amount columns.
    columns = [table[name].tolist() for name in columns]
    result = tuple(tuple(row) for row in zip(*columns, strict=True))
    encoded = tuple(_json(_coordinate(key)) for key in result)
    _require(
        not unique or len(set(encoded)) == len(encoded),
        "DUPLICATE_ORIGINAL_COORDINATES",
    )
    return result


def _metadata(model, *, source_sha256, support_json):
    document = _document(model.to_bytes(), empirical.MAX_MODEL_BYTES)
    _require(
        document["weight_kind"] == "explicit"
        and _json(document["support"]).decode() == support_json,
        "MODEL_CONTRACT",
    )
    return dict(
        protocol=PROTOCOL + "/model-metadata",
        model_sha256=model.sha256,
        support_sha256=_sha(support_json.encode()),
        donor_source_sha256=source_sha256,
        weight_kind="explicit",
        targets=document["targets"],
        diagnostics=model.diagnostics,
    )


def _fit_expected(context):
    p = dict(context.params)
    support = _support(p.pop("support_json"))
    _require(p.pop("weight_kind") == "explicit", "WEIGHT_KIND")
    p.pop("support_sha256")
    return joint_empirical_fit_node(
        context.node.id,
        population=context.node.population,
        source_projection=context.node.artifact_inputs[0],
        support=support,
        **p,
    )


def _draw_expected(context):
    p = dict(context.params)
    support, transport = (
        _support(p.pop("support_json")),
        _transport(p.pop("transport_json")),
    )
    p.pop("support_sha256")
    p.pop("transport_sha256")
    edges = context.node.artifact_inputs
    _require(
        len(edges) == 3 and edges[0].producer == edges[1].producer, "MODEL_PRODUCER"
    )
    return joint_empirical_draw_node(
        context.node.id,
        population=context.node.population,
        model_producer=edges[0].producer,
        source_projection=edges[2],
        support=support,
        transport=transport,
        **p,
    )


class JointEmpiricalFitKernel(KernelBase):
    ref = "fit.joint_empirical.fit@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            empirical,
            weight_model,
            randomness,
            canonical,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        _require(
            dict(context.params) == dict(context.node.params), "CONTEXT_PARAMETERS"
        )
        _require(len(context.node.artifact_inputs) == 1, "FIT_ARTIFACTS")
        expected = _fit_expected(context)
        _require(context.node.normative() == expected.normative(), "DECLARATION")
        table = _table(context)
        p = context.params
        for column in (*p["targets"], p["weight_column"]):
            _require(
                table[column].dtype == np.dtype("float64"), "FLOAT64_COLUMN:" + column
            )
        model = empirical.fit_joint_empirical(
            table.loc[:, list(p["targets"])].copy(deep=True),
            targets=p["targets"],
            donor_keys=_keys(table, p["donor_key_columns"], unique=True),
            household_keys=_keys(table, p["household_key_columns"], unique=False),
            support=_support(p["support_json"]),
            weights=table[p["weight_column"]].to_numpy(copy=True),
        )
        metadata = _metadata(
            model, source_sha256=p["source_sha256"], support_json=p["support_json"]
        )
        return KernelResult(
            artifacts={"model": model.to_bytes(), "model_metadata": _json(metadata)},
            receipt=metadata,
        )


def _draw_document(draw, ids, keys, p):
    rows = [
        dict(
            support_id=str(int(identity)),
            coordinate=_coordinate(key),
            values=draw.values[i].tolist(),
            pattern=int(draw.patterns[i]),
            donor_key=None
            if draw.donor_keys[i] is None
            else _coordinate(draw.donor_keys[i]),
        )
        for i, (identity, key) in enumerate(zip(ids, keys, strict=True))
    ]
    rows.sort(key=lambda row: _json(row["coordinate"]))
    return dict(
        protocol=PROTOCOL + "/draw",
        model_sha256=draw.model_sha256,
        support_sha256=p["support_sha256"],
        donor_source_sha256=p["donor_source_sha256"],
        source_sha256=p["source_sha256"],
        scenario_sha256=p["scenario_sha256"],
        transport_json=p["transport_json"],
        transport_sha256=p["transport_sha256"],
        stream=list(p["stream"]),
        coordinate_suffix=_coordinate(p["coordinate_suffix"]),
        rows=rows,
    )


class JointEmpiricalDrawKernel(KernelBase):
    ref = "fit.joint_empirical.draw@1"
    capabilities = Capabilities(
        Determinism.SEEDED,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.KEYED,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            empirical,
            weight_model,
            randomness,
            canonical,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        _require(
            dict(context.params) == dict(context.node.params), "CONTEXT_PARAMETERS"
        )
        expected = _draw_expected(context)
        _require(context.node.normative() == expected.normative(), "DECLARATION")
        table = _table(context)
        p = context.params
        _require(len(table) <= MAX_RECIPIENTS, "RECIPIENT_COUNT")
        model_value = _artifact(context, "model", platform=True)
        metadata_value = _artifact(context, "model_metadata", platform=True)
        _require(
            model_value.producer_key == metadata_value.producer_key, "MODEL_PRODUCER"
        )
        model = empirical.JointEmpiricalModel.from_bytes(model_value.payload)
        metadata = _metadata(
            model,
            source_sha256=p["donor_source_sha256"],
            support_json=p["support_json"],
        )
        _document(metadata_value.payload)
        _require(metadata_value.payload == _json(metadata), "MODEL_METADATA")
        # Keep every real original recipient in the support Frame. The graph
        # executor cannot construct typed empty weights for an all-false row
        # Slice; this explicit boolean input selects draws within the kernel.
        candidate_rows = len(table)
        eligibility = p["eligibility_column"]
        if eligibility is not None:
            _require(table[eligibility].dtype == np.dtype("bool"), "ELIGIBILITY_BOOL")
            _keys(table, p["recipient_key_columns"], unique=True)
            table = table.loc[table[eligibility]]
        keys = _keys(table, p["recipient_key_columns"], unique=True)
        uniforms = {
            purpose: randomness.keyed_uniform(
                stream=p["stream"],
                keys=[(*key, *p["coordinate_suffix"], purpose) for key in keys],
            )
            for purpose in ("pattern", "donor")
        }
        draw = empirical.draw_joint_empirical(
            model,
            pattern_uniforms=uniforms["pattern"],
            donor_uniforms=uniforms["donor"],
            transport=_transport(p["transport_json"]),
        )
        document = _draw_document(draw, table[p["entity"] + "_id"], keys, p)
        payload = _json(document)
        _require(len(payload) <= MAX_DRAW_BYTES, "DRAW_BYTES")
        return KernelResult(
            artifacts={"draws": payload},
            receipt=dict(
                protocol=PROTOCOL,
                rows=len(keys),
                candidate_rows=candidate_rows,
                model_sha256=model.sha256,
                support_sha256=p["support_sha256"],
                donor_source_sha256=p["donor_source_sha256"],
                source_sha256=p["source_sha256"],
                scenario_sha256=p["scenario_sha256"],
                transport_sha256=p["transport_sha256"],
                draw_sha256=_sha(payload),
            ),
        )


def read_joint_empirical_draw(
    payload,
    *,
    recipient_ids,
    recipient_keys,
    model_sha256,
    support,
    donor_source_sha256,
    source_sha256,
    scenario_sha256,
    transport,
    stream,
    coordinate_suffix,
):
    """Validate private shape/identity bindings, not the empirical draw law.

    The host must independently re-execute the actual model/stream from its
    retained qualified source before treating these values as verified output.
    A well-formed detached artifact is descriptive, not source/host authority.
    """
    _require(
        type(recipient_ids) is pd.Index
        and recipient_ids.dtype == np.dtype("int64")
        and recipient_ids.is_unique,
        "RECIPIENT_INDEX",
    )
    _require(
        len(recipient_ids) == len(recipient_keys) <= MAX_RECIPIENTS, "RECIPIENT_COUNT"
    )
    _stream(stream, coordinate_suffix)
    _require(
        type(support) is empirical.SupportRequirements
        and type(transport) is empirical.JointTransport,
        "READER_PARAMETERS",
    )
    _require(
        all(
            _hash(v)
            for v in (model_sha256, donor_source_sha256, source_sha256, scenario_sha256)
        ),
        "READER_HASHES",
    )
    doc = _document(payload)
    expected = dict(
        protocol=PROTOCOL + "/draw",
        model_sha256=model_sha256,
        support_sha256=_sha(_json(support.document())),
        donor_source_sha256=donor_source_sha256,
        source_sha256=source_sha256,
        scenario_sha256=scenario_sha256,
        transport_json=_json(transport.document()).decode(),
        transport_sha256=_sha(_json(transport.document())),
        stream=list(stream),
        coordinate_suffix=_coordinate(coordinate_suffix),
    )
    _require(
        set(doc) == {*expected, "rows"}
        and _json({k: doc[k] for k in expected}) == _json(expected),
        "DRAW_BINDING",
    )
    rows = doc["rows"]
    _require(type(rows) is list and len(rows) == len(recipient_ids), "DRAW_ROWS")
    wanted = {
        _json(_coordinate(key)): (i, str(int(identity)))
        for i, (identity, key) in enumerate(
            zip(recipient_ids, recipient_keys, strict=True)
        )
    }
    _require(len(wanted) == len(recipient_ids), "DUPLICATE_ORIGINAL_COORDINATES")
    ordered, values, patterns, donors = (
        [],
        np.zeros((len(rows), 2)),
        np.zeros(len(rows), dtype=np.int8),
        [None] * len(rows),
    )
    for row in rows:
        _require(
            type(row) is dict
            and set(row)
            == {"support_id", "coordinate", "values", "pattern", "donor_key"},
            "DRAW_ROW_SCHEMA",
        )
        key = empirical._decode_key(row["coordinate"])
        encoded = _json(_coordinate(key))
        _require(encoded in wanted, "DRAW_COORDINATE")
        i, identity = wanted[encoded]
        _require(row["support_id"] == identity, "DRAW_SUPPORT_ID")
        ordered.append(encoded)
        pair = row["values"]
        _require(
            type(pair) is list
            and len(pair) == 2
            and all(type(v) is float and np.isfinite(v) and v >= 0 for v in pair),
            "DRAW_VALUES",
        )
        pattern = row["pattern"]
        _require(
            type(pattern) is int and pattern == int(pair[0] > 0) + 2 * int(pair[1] > 0),
            "DRAW_PATTERN",
        )
        donor = (
            None
            if row["donor_key"] is None
            else empirical._decode_key(row["donor_key"])
        )
        _require((donor is None) == (pattern == 0), "DRAW_DONOR")
        values[i], patterns[i], donors[i] = pair, pattern, donor
    _require(ordered == sorted(wanted), "DRAW_ORDER_OR_DUPLICATE")
    frozen = np.frombuffer(values.tobytes(), dtype=np.float64).reshape((-1, 2))
    return empirical.JointEmpiricalDraw(
        frozen,
        np.frombuffer(patterns.tobytes(), dtype=np.int8),
        tuple(donors),
        model_sha256,
        _json(transport.document()),
    )
