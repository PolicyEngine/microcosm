"""The shared contract between transport evidence producers and their gates.

Three release-blocking gates (``support``, ``release_input_coverage`` and
``weights_audit``) read typed artifacts that graph nodes produce
(:mod:`~microcosm.build.transport.evidence_kernels`). This leaf module holds
what both sides need: the artifact types and their decoders, the identity
header every evidence artifact carries, the per-column frame digest, the
implementation hash of each evidence kernel computed by module name, the
closure partition, and a reader for a binding's relation declarations.

It imports neither :mod:`~microcosm.build.transport.registry` nor any kernel
module, not even for type hints: ``gates.battery@1`` hashes the source
closure of its bindings (:mod:`~microcosm.build.transport.binding_identity`),
which follows every import statement, so an import of a kernel module here
would re-key the battery on any transport edit. Kernel modules enter only as
module *names*, which that closure does not follow.

**Implementation hashes by name.** :func:`evidence_implementation_hash`
reproduces :func:`microcosm.graph.source_hash` over each evidence kernel's
declared module list (:data:`EVIDENCE_KERNEL_MODULES`) in list order, from
module stubs whose ``__file__`` comes from :func:`importlib.util.find_spec`.
It never imports the kernel module. Each evidence kernel's own
``implementation_hash`` calls the same function, so the producer's node key
and the battery's expected hash cannot differ. Every list starts with the
kernel's own module, so an edit to ``run`` moves both. No adapter class
enters the hash; engine identity enters a producer's node key through its
``{"prepared": "engine_refs"}`` parameter.

**The header.** Every evidence artifact is canonical JSON::

    {"schema_version": 1, "kind": <kind>,
     "producer": {"kernel": <ref>, "implementation_hash": <sha256>,
                  "node": <node id>},
     "inputs": {"artifacts": {<alias>: <payload sha256>},
                "frame": {"<entity>.<column>": <column sha256>},
                "params": {<name>: <sha256 of the canonical param value>}},
     ...body}

:func:`checked_header` binds those claims to the battery that reads them,
through the battery's reserved ``graph_evidence`` alias: the producer node is
the one the battery's own edge names, the kernel and hash are the expected
ones, every upstream artifact digest equals the digest of the bytes the
battery received from that producer, and every frame column the battery also
slices digests to the same value. A forged, stale or substituted artifact
therefore fails its gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame.adapters.axiom_input_surface import (
    AxiomInputSurface,
    AxiomModuleSurface,
)
from microcosm.frame.concept_mapping import ConceptMapping, InputRef
from microcosm.frame.concepts import CONCEPTS
from microcosm.frame.input_closure import (
    ClosureClass,
    EncodedBy,
    InputClosure,
    KnobKind,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.decl import ArtifactType
from microcosm.graph.kernel import source_hash

__all__ = [
    "AXIOM_INPUT_CLOSURE",
    "BINDING_LOCAL_VIEW_MESSAGE",
    "CLOSURE_CLASSES",
    "DONOR_ARTIFACT_RECEIPT",
    "DONOR_RECEIPT_REF",
    "DONOR_RECEIPT_TYPE",
    "EVIDENCE_KERNEL_DEPENDENCIES",
    "EVIDENCE_KERNEL_MODULES",
    "EVIDENCE_TYPES",
    "EvidenceDocument",
    "FIT_WEIGHT_RECORDS",
    "DONOR_RANGES",
    "DONOR_RANGES_REF",
    "DONOR_RANGES_TYPE",
    "DONOR_SUPPORT_BOUNDS",
    "SUPPORT_BOUNDS_REF",
    "SUPPORT_BOUNDS_TYPE",
    "SUPPORT_ROW_KEYS",
    "decode_donor_ranges",
    "decode_support_bounds",
    "inside_quantile_support",
    "quantile_bounds",
    "quantile_support",
    "WEIGHTS_AUDIT_REF",
    "WEIGHTS_AUDIT_TYPE",
    "decode_weights_audit",
    "GRAPH_EVIDENCE",
    "INPUT_CLOSURE_REF",
    "INPUT_CLOSURE_TYPE",
    "RelationInput",
    "binding_relation_inputs",
    "checked_header",
    "closure_partition",
    "decode_donor_receipt",
    "decode_input_closure",
    "evidence_header",
    "evidence_implementation_hash",
    "frame_column_sha256",
    "module_declares_imports",
    "param_sha256",
    "partition_counts",
    "payload_sha256",
    "unreached_refs",
    "without_unreached",
]

#: The battery's reserved alias: one row per declared artifact edge.
GRAPH_EVIDENCE = "graph_evidence"
#: Aliases a battery binding reads; none may be a ``gates.json`` parameter.
DONOR_ARTIFACT_RECEIPT = "donor_artifact_receipt"
AXIOM_INPUT_CLOSURE = "axiom_input_closure"

FIT_WEIGHT_RECORDS = "fit_weight_records"

DONOR_RECEIPT_REF = "transport.donor_receipt@1"
INPUT_CLOSURE_REF = "transport.input_closure@1"
WEIGHTS_AUDIT_REF = "transport.weights_audit@1"
DONOR_RANGES_REF = "transport.donor_ranges@1"
SUPPORT_BOUNDS_REF = "transport.support_bounds@1"
DONOR_RANGES = "donor_ranges"
DONOR_SUPPORT_BOUNDS = "donor_support_bounds"

DONOR_RECEIPT_TYPE = ArtifactType("microcosm.transport.donor-artifact-receipt", 1)
INPUT_CLOSURE_TYPE = ArtifactType("microcosm.transport.axiom-input-closure", 1)
WEIGHTS_AUDIT_TYPE = ArtifactType("microcosm.transport.weights-audit", 1)
DONOR_RANGES_TYPE = ArtifactType("microcosm.transport.donor-ranges", 1)
SUPPORT_BOUNDS_TYPE = ArtifactType("microcosm.transport.donor-support-bounds", 1)

_DONOR_RECEIPT_KIND = "transport_donor_artifact_receipt"
_INPUT_CLOSURE_KIND = "transport_axiom_input_closure"
_WEIGHTS_AUDIT_KIND = "transport_weights_audit"
_DONOR_RANGES_KIND = "transport_donor_ranges"
_SUPPORT_BOUNDS_KIND = "transport_donor_support_bounds"

_KERNELS = "microcosm.build.transport.evidence_kernels"
_LEAF = __name__
_CANONICAL = "microcosm.graph.canonical"

#: Each evidence kernel's source modules, in ``source_hash`` order. Each list
#: starts with the kernel's own module and ends with this leaf.
EVIDENCE_KERNEL_MODULES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        DONOR_RECEIPT_REF: (_KERNELS, _CANONICAL, _LEAF),
        INPUT_CLOSURE_REF: (
            _KERNELS,
            "microcosm.frame.input_closure",
            "microcosm.frame.concept_mapping",
            "microcosm.frame.concepts",
            "microcosm.frame.adapters.axiom_input_surface",
            "microcosm.frame.adapters.axiom",
            _CANONICAL,
            _LEAF,
        ),
        WEIGHTS_AUDIT_REF: (
            _KERNELS,
            "microcosm.build.transport.weight_quality",
            "microcosm.calibrate.artifacts",
            "microcosm.calibrate.matrix",
            "microcosm.calibrate.registry",
            "microcosm.calibrate.solve",
            _CANONICAL,
            _LEAF,
        ),
        DONOR_RANGES_REF: (_KERNELS, _CANONICAL, _LEAF),
        SUPPORT_BOUNDS_REF: (
            _KERNELS,
            "microcosm.build.transport.money_kernels",
            "microcosm.build.money_normalization",
            "microcosm.build.reference_factors",
            "microcosm.build.ledger_artifact",
            "microcosm.build.ledger_targets",
            "microcosm.build.chronicle_epoch",
            "microcosm.build.transport.graph_inputs",
            "microcosm.frame.concepts",
            "microcosm.frame.exact_money",
            "microcosm.frame.transport",
            _CANONICAL,
            _LEAF,
        ),
    }
)
#: Each evidence kernel's declared third-party dependencies.
EVIDENCE_KERNEL_DEPENDENCIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {ref: ("numpy", "pandas") for ref in EVIDENCE_KERNEL_MODULES}
)

#: The report classes a closure row partitions its live roots into.
CLOSURE_CLASSES = (
    "encoded",
    "knob_targeted",
    "defaulted",
    "engine_optional",
    "relation",
    "unreached",
    "gap",
)

BINDING_LOCAL_VIEW_MESSAGE = (
    "{module} {entity}.{input}: closure source 'binding_local_view' is reserved "
    "for binding-local views (S2b, S2D-AGREED item 3) and is refused until that "
    "extension lands."
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMPORTS = re.compile(rb"^imports\s*:", re.MULTILINE)


# ---------------------------------------------------------------------------
# Implementation hashes by module name
# ---------------------------------------------------------------------------


def evidence_implementation_hash(ref: str) -> str:
    """``source_hash`` over ``ref``'s declared modules, without importing them.

    Each name resolves through :func:`importlib.util.find_spec`, which imports
    only parent packages; the stub carries the module's name and source path,
    which is all ``source_hash`` reads. The order is the declared order.
    """

    try:
        names = EVIDENCE_KERNEL_MODULES[ref]
    except KeyError:
        raise ValueError(f"{ref!r} is not a transport evidence kernel.") from None
    stubs = []
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not spec.has_location:
            raise ValueError(
                f"Evidence kernel {ref} lists {name!r}, which has no source."
            )
        stub = types.ModuleType(name)
        stub.__file__ = spec.origin
        stubs.append(stub)
    return source_hash(*stubs, dependencies=EVIDENCE_KERNEL_DEPENDENCIES[ref])


# ---------------------------------------------------------------------------
# Digests and the header
# ---------------------------------------------------------------------------


def payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def param_sha256(value: object) -> str:
    """SHA-256 of a parameter value's canonical JSON."""

    return hashlib.sha256(canonical_json(_plain_param(value))).hexdigest()


def _plain_param(value: object) -> object:
    if isinstance(value, tuple | list):
        return [_plain_param(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain_param(item) for key, item in value.items()}
    return value


def _array_bytes(values: object) -> bytes:
    """Deterministic bytes of one column in a fixed row order."""

    series = pd.Series(values)
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
        array = np.ascontiguousarray(series.to_numpy())
        return dtype.str.encode("ascii") + b"\0" + array.tobytes()
    items = []
    for item in series.to_numpy(dtype=object):
        if item is None or (isinstance(item, float) and math.isnan(item)):
            items.append(None)
        elif item is pd.NA or item is pd.NaT:
            items.append(None)
        elif isinstance(item, np.generic):
            items.append(item.item())
        else:
            items.append(item)
    for index, item in enumerate(items):
        if isinstance(item, float) and not math.isfinite(item):
            items[index] = repr(item)
    return str(dtype).encode("utf-8") + b"\0" + canonical_json(items)


def frame_column_sha256(entity: str, column: str, ids: object, values: object) -> str:
    """One column's digest, in ascending entity-id order.

    Producer and battery compute it the same way from their own slices, so a
    digest that differs names a column whose values differ.
    """

    id_values = pd.Series(ids).to_numpy()
    column_values = pd.Series(values).reset_index(drop=True)
    if len(id_values) != len(column_values):
        raise ValueError(f"{entity}.{column}: ids and values differ in length.")
    order = np.argsort(id_values, kind="stable")
    digest = hashlib.sha256(b"microcosm.transport.frame-column/1\0")
    digest.update(f"{entity}\0{column}\0".encode())
    digest.update(_array_bytes(id_values[order]))
    digest.update(b"\0")
    digest.update(_array_bytes(column_values.iloc[order].reset_index(drop=True)))
    return digest.hexdigest()


def evidence_header(
    kind: str,
    *,
    kernel: str,
    node: str,
    artifacts: Mapping[str, bytes],
    frame: Mapping[str, str],
    params: Mapping[str, object],
) -> dict[str, object]:
    """The identity header of one evidence artifact (module docstring)."""

    return {
        "schema_version": 1,
        "kind": kind,
        "producer": {
            "kernel": kernel,
            "implementation_hash": evidence_implementation_hash(kernel),
            "node": node,
        },
        "inputs": {
            "artifacts": {
                alias: payload_sha256(payload)
                for alias, payload in sorted(artifacts.items())
            },
            "frame": dict(sorted(frame.items())),
            "params": {
                name: param_sha256(value) for name, value in sorted(params.items())
            },
        },
    }


@dataclass(frozen=True)
class EvidenceDocument:
    """A decoded evidence artifact: its canonical document and payload digest."""

    kind: str
    sha256: str
    document: Mapping[str, object]


def _decode(payload: bytes, *, kind: str, body: frozenset[str]) -> EvidenceDocument:
    if type(payload) is not bytes:
        raise TypeError("Evidence is immutable bytes.")
    document = json.loads(payload)
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise ValueError(f"A {kind} artifact must be canonical JSON.")
    header = {"schema_version", "kind", "producer", "inputs"}
    if set(document) != header | body:
        raise ValueError(
            f"A {kind} artifact holds exactly {sorted(header | body)}; got "
            f"{sorted(document)}."
        )
    if document["schema_version"] != 1 or document["kind"] != kind:
        raise ValueError(f"Unsupported {kind} artifact.")
    producer, inputs = document["producer"], document["inputs"]
    if (
        not isinstance(producer, dict)
        or set(producer) != {"kernel", "implementation_hash", "node"}
        or not all(isinstance(producer[key], str) for key in producer)
        or not isinstance(inputs, dict)
        or set(inputs) != {"artifacts", "frame", "params"}
        or not all(
            isinstance(inputs[key], dict)
            and all(
                isinstance(name, str)
                and isinstance(value, str)
                and _SHA256.fullmatch(value)
                for name, value in inputs[key].items()
            )
            for key in inputs
        )
    ):
        raise ValueError(f"A {kind} artifact's header is malformed.")
    return EvidenceDocument(
        kind=kind,
        sha256=payload_sha256(payload),
        document=MappingProxyType(document),
    )


def decode_donor_receipt(payload: bytes) -> EvidenceDocument:
    """Decode a ``donor_artifact_receipt``; its body is checked by the gate."""

    decoded = _decode(
        payload,
        kind=_DONOR_RECEIPT_KIND,
        body=frozenset({"donor", "pin", "content_basis", "donor_time_period"}),
    )
    document = decoded.document
    donor, pin = document["donor"], document["pin"]
    if (
        not isinstance(donor, dict)
        or set(donor) != {"sha256", "size_bytes", "locator", "revision"}
        or not isinstance(pin, dict)
        or set(pin) != {"stage", "artifact_index", "sha256", "size_bytes"}
    ):
        raise ValueError("A donor receipt's donor and pin records are malformed.")
    return decoded


def decode_input_closure(payload: bytes) -> EvidenceDocument:
    """Decode an ``axiom_input_closure``; its rows are checked by the gate."""

    decoded = _decode(
        payload,
        kind=_INPUT_CLOSURE_KIND,
        body=frozenset(
            {"closure_sha256", "mapping_sha256", "engine", "check", "modules"}
        ),
    )
    document = decoded.document
    if not isinstance(document["modules"], list) or not all(
        isinstance(row, dict) and set(row) == _MODULE_ROW_KEYS
        for row in document["modules"]
    ):
        raise ValueError("An input closure artifact's module rows are malformed.")
    if not isinstance(document["check"], list) or not isinstance(
        document["engine"], dict
    ):
        raise ValueError("An input closure artifact's check or engine is malformed.")
    return decoded


_WEIGHTS_AUDIT_BODY = frozenset(
    {
        "weight_entity",
        "fit_records",
        "transition",
        "binding",
        "mass",
        "source_households",
        "partitions",
    }
)


def decode_weights_audit(payload: bytes) -> EvidenceDocument:
    """Decode a ``fit_weight_records`` audit; its rows are checked by the gate."""

    decoded = _decode(payload, kind=_WEIGHTS_AUDIT_KIND, body=_WEIGHTS_AUDIT_BODY)
    document = decoded.document
    transition = document["transition"]
    if (
        not isinstance(document["fit_records"], list)
        or not isinstance(document["partitions"], list)
        or not isinstance(transition, dict)
        or set(transition)
        != {
            "from_kind",
            "to_kind",
            "household_id",
            "source_household_id",
            "design_weight",
            "calibrated_weight",
        }
        or len(
            {
                len(transition[key])
                for key in (
                    "household_id",
                    "source_household_id",
                    "design_weight",
                    "calibrated_weight",
                )
            }
        )
        != 1
    ):
        raise ValueError("A weights audit's records are malformed.")
    return decoded


def decode_donor_ranges(payload: bytes) -> EvidenceDocument:
    """Decode the donor's realized ranges; its rows are checked by the gate."""

    decoded = _decode(payload, kind=_DONOR_RANGES_KIND, body=frozenset({"columns"}))
    if not isinstance(decoded.document["columns"], list):
        raise ValueError("A donor-ranges artifact's rows are malformed.")
    return decoded


#: Support rows hold exactly these fields.
SUPPORT_ROW_KEYS = frozenset(
    {
        "entity",
        "column",
        "class",
        "owner",
        "realized_min",
        "realized_max",
        "expected_min",
        "expected_max",
        "expected_categories",
        "records_outside",
        "weighted_outside",
        "sources",
        "reason",
    }
)


def decode_support_bounds(payload: bytes) -> EvidenceDocument:
    """Decode a ``donor_support_bounds`` artifact; the gate checks its rows."""

    decoded = _decode(
        payload,
        kind=_SUPPORT_BOUNDS_KIND,
        body=frozenset({"rate", "structural_columns", "columns"}),
    )
    rows = decoded.document["columns"]
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) and set(row) == SUPPORT_ROW_KEYS for row in rows
    ):
        raise ValueError("A donor-support artifact's rows are malformed.")
    return decoded


#: Artifact type -> decoder, for ``gates.battery@1``.
EVIDENCE_TYPES: Mapping[ArtifactType, object] = MappingProxyType(
    {
        DONOR_RECEIPT_TYPE: decode_donor_receipt,
        INPUT_CLOSURE_TYPE: decode_input_closure,
        WEIGHTS_AUDIT_TYPE: decode_weights_audit,
        DONOR_RANGES_TYPE: decode_donor_ranges,
        SUPPORT_BOUNDS_TYPE: decode_support_bounds,
    }
)

_KINDS = {
    DONOR_RECEIPT_REF: (_DONOR_RECEIPT_KIND, DONOR_RECEIPT_TYPE),
    INPUT_CLOSURE_REF: (_INPUT_CLOSURE_KIND, INPUT_CLOSURE_TYPE),
    WEIGHTS_AUDIT_REF: (_WEIGHTS_AUDIT_KIND, WEIGHTS_AUDIT_TYPE),
    DONOR_RANGES_REF: (_DONOR_RANGES_KIND, DONOR_RANGES_TYPE),
    SUPPORT_BOUNDS_REF: (_SUPPORT_BOUNDS_KIND, SUPPORT_BOUNDS_TYPE),
}


def quantile_support(
    bands: object, interpolation: object
) -> tuple[list[tuple[float, float | None]], list[tuple[float, float | None]]]:
    """The magnitude intervals a quantile map can write, by sign.

    ``bands`` is the declared band document the quantile map reads:
    ``{bands: rows}``, ``{positive: rows, negative: rows}`` or
    ``{components: [{value, bands: rows}]}``, where rows are ``{lower, upper,
    reference}`` over positive magnitudes (a bare row list is read as
    ``bands``). ``uniform`` needs finite uppers; a
    ``pareto`` top band may have no upper bound, recorded as ``None``. Any
    other mode is refused by name. Zero always maps to zero.
    """

    method = (
        interpolation.get("method")
        if isinstance(interpolation, Mapping)
        else interpolation
    )
    if method not in {"uniform", "pareto"}:
        raise ValueError(
            f"Support handles uniform and top-band pareto quantile maps only; "
            f"{method!r} lands with its own producer."
        )

    def rows(value) -> list[tuple[float, float | None]]:
        if not isinstance(value, list) or not value:
            raise ValueError("Quantile support needs a nonempty band list.")
        out = []
        for index, row in enumerate(value):
            upper = row.get("upper")
            if upper is None and (method != "pareto" or index != len(value) - 1):
                raise ValueError("Only a pareto top band may have no upper bound.")
            out.append((float(row["lower"]), None if upper is None else float(upper)))
        return out

    positive: list = []
    negative: list = []
    if isinstance(bands, Mapping) and set(bands) == {"bands"}:
        bands = bands["bands"]
    if isinstance(bands, Mapping) and set(bands) == {"components"}:
        components = bands["components"]
        documents = (
            [item["bands"] for item in components]
            if isinstance(components, list)
            else list(components.values())
        )
    else:
        documents = [bands]
    for document in documents:
        if isinstance(document, Mapping) and set(document) == {"bands"}:
            document = document["bands"]
        if isinstance(document, Mapping):
            if not set(document) <= {"positive", "negative"}:
                raise ValueError(
                    "Quantile bands declare bands, positive and negative rows, or "
                    "components."
                )
            positive += rows(document["positive"]) if "positive" in document else []
            negative += rows(document["negative"]) if "negative" in document else []
        else:
            positive += rows(document)
    return positive, negative


def quantile_bounds(
    positive: Sequence[tuple[float, float | None]],
    negative: Sequence[tuple[float, float | None]],
) -> tuple[float | None, float | None]:
    """``[expected_min, expected_max]`` of a quantile map's support; ``None`` is open."""

    def top(intervals):
        return (
            None
            if any(u is None for _, u in intervals)
            else max(u for _, u in intervals)
        )

    maximum = top(positive) if positive else float(0)
    minimum = (None if top(negative) is None else -top(negative)) if negative else float(0)
    return minimum, maximum


def inside_quantile_support(
    value: float,
    positive: Sequence[tuple[float, float | None]],
    negative: Sequence[tuple[float, float | None]],
) -> bool:
    """Whether one mapped value lies in the declared bands or is zero."""

    if value == 0:
        return True
    intervals = positive if value > 0 else negative
    magnitude = abs(value)
    return any(
        lower <= magnitude and (upper is None or magnitude <= upper)
        for lower, upper in intervals
    )


def checked_header(
    alias: str,
    evidence: object,
    graph_evidence: object,
    frame: object,
    *,
    kernel: str,
) -> list[str]:
    """Every way ``evidence`` fails to be the artifact this battery declared.

    Returns failure lines in a fixed order; empty when the header holds:

    1. ``evidence`` decodes to the kernel's kind, and the bytes are the ones
       the battery received under ``alias`` (``graph_evidence``).
    2. ``producer.node`` is the producer the battery's edge names, and the
       edge's type is the kernel's type.
    3. ``producer.kernel`` is ``kernel`` and its implementation hash is the
       one this process computes for it.
    4. Every upstream artifact digest equals the digest of the bytes the
       battery received under the same alias.
    5. Every frame column the battery's own frame also carries digests to
       the same value.
    """

    kind, artifact_type = _KINDS[kernel]
    label = f"{alias}"
    if not isinstance(evidence, EvidenceDocument) or evidence.kind != kind:
        return [f"{label}: evidence is not a decoded {kind} artifact."]
    if not isinstance(graph_evidence, Mapping) or alias not in graph_evidence:
        return [f"{label}: the battery declares no artifact edge for this alias."]
    edges = graph_evidence
    edge = edges[alias]
    document = evidence.document
    producer = document["producer"]
    failures: list[str] = []
    if edge.get("sha256") != evidence.sha256:
        failures.append(
            f"{label}: payload SHA-256 {evidence.sha256} is not the "
            f"{edge.get('sha256')} the battery received."
        )
    if list(edge.get("type", ())) != [artifact_type.name, artifact_type.schema_version]:
        failures.append(
            f"{label}: the battery's edge is not typed {artifact_type.name}."
        )
    if producer["node"] != edge.get("producer"):
        failures.append(
            f"{label}: produced by node {producer['node']!r}, but the battery's "
            f"edge names {edge.get('producer')!r}."
        )
    if producer["kernel"] != kernel:
        failures.append(f"{label}: produced by {producer['kernel']!r}, not {kernel!r}.")
    expected = evidence_implementation_hash(kernel)
    if producer["implementation_hash"] != expected:
        failures.append(
            f"{label}: implementation hash {producer['implementation_hash']} is "
            f"not this build's {expected}."
        )
    for name, digest in sorted(document["inputs"]["artifacts"].items()):
        other = edges.get(name)
        if not isinstance(other, Mapping):
            failures.append(
                f"{label}: input artifact {name!r} is not declared on the battery."
            )
        elif other.get("sha256") != digest:
            failures.append(
                f"{label}: input artifact {name!r} digest {digest} is not the "
                f"{other.get('sha256')} the battery received."
            )
    if frame is not None:
        for key, digest in sorted(document["inputs"]["frame"].items()):
            entity, _, column = key.partition(".")
            if entity not in frame.entities:
                continue
            table = frame.table(entity)
            if column not in table.columns:
                continue
            ids = table[frame.schema.entity_id_column(entity)]
            if frame_column_sha256(entity, column, ids, table[column]) != digest:
                failures.append(
                    f"{label}: frame column {key} differs from the battery's frame."
                )
    return failures


# ---------------------------------------------------------------------------
# Relations and binding declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class RelationInput:
    """One relation input: the engine name, its physical column and role."""

    input: str
    column: str
    role_column: str | None = None


def binding_relation_inputs(row: Mapping[str, object]) -> tuple[RelationInput, ...]:
    """The related inputs a binding's ``relations`` block declares, sorted.

    Reads the manifest fields only; the registry validates the manifests
    themselves (``registry._binding_relations``), and a differential test
    holds this reader to it.
    """

    relations = row.get("relations")
    if relations is None:
        return ()
    if isinstance(relations, str | bytes) or not isinstance(relations, Sequence):
        raise TypeError(f"rules binding {row.get('id')!r}.relations must be a list.")
    inputs: dict[str, RelationInput] = {}
    for manifest in relations:
        if not isinstance(manifest, Mapping) or not isinstance(
            manifest.get("related_inputs"), Mapping
        ):
            raise ValueError(
                f"rules binding {row.get('id')!r} relation manifest lacks related_inputs."
            )
        role = manifest.get("role_column")
        for name, column in manifest["related_inputs"].items():
            item = RelationInput(
                str(name), str(column), None if role is None else str(role)
            )
            if inputs.get(item.input, item) != item:
                raise ValueError(
                    f"rules binding {row.get('id')!r} relates {name!r} twice."
                )
            inputs[item.input] = item
    return tuple(sorted(inputs.values()))


def module_declares_imports(source: bytes) -> bool:
    """Whether a RuleSpec module has a top-level ``imports:`` key."""

    return _IMPORTS.search(source) is not None


# ---------------------------------------------------------------------------
# Unreached roots and the closure partition
# ---------------------------------------------------------------------------


def unreached_refs(bindings: Iterable[Mapping[str, object]]) -> frozenset[InputRef]:
    """The roots each binding's ``unreached_roots`` declares, as module refs."""

    return frozenset(
        InputRef(str(name), str(row["engine_entity"]), str(row["rulespec_path"]))
        for row in bindings
        if isinstance(row.get("unreached_roots"), Mapping)
        for name in row["unreached_roots"].get("names", ())
    )


def without_unreached(
    surface: AxiomInputSurface,
    mapping: ConceptMapping,
    unreached: Iterable[InputRef],
    mapped_entities: Iterable[str],
) -> tuple[AxiomInputSurface, ConceptMapping]:
    """The surface and mapping ``InputClosure.check`` reads once roots are excused.

    Drops each declared unreached ref, and every root on an engine entity the
    schema does not map, from the surface; drops the bindings of the declared
    refs from the mapping and lists every concept no remaining binding reads
    as unmapped (``ConceptMapping`` refuses a concept left neither bound nor
    unmapped).
    """

    excused = set(unreached)
    mapped = set(mapped_entities)
    modules = {}
    for path, module in surface.modules.items():
        inputs = {
            entity: tuple(
                name for name in names if InputRef(name, entity, path) not in excused
            )
            for entity, names in module.inputs.items()
            if entity in mapped
        }
        modules[path] = replace(module, inputs=MappingProxyType(inputs))
    bindings = tuple(
        binding for binding in mapping.bindings if binding.ref not in excused
    )
    bound = {concept for binding in bindings for concept in binding.reads}
    unmapped = {
        item.id: mapping.unmapped.get(
            item.id, "Every binding that read it reads an unreached root."
        )
        for item in CONCEPTS
        if item.id not in bound
    }
    return (
        replace(surface, modules=MappingProxyType(modules)),
        replace(mapping, bindings=bindings, unmapped=unmapped),
    )


_MODULE_ROW_KEYS = frozenset(
    {
        "binding",
        "rulespec_path",
        "sha256",
        "reviewed_sha256",
        "engine_entity",
        "roots",
        "unmapped_entity",
        "encoded",
        "knob_targeted",
        "defaulted",
        "engine_optional",
        "relation",
        "unreached",
        "gap",
        "unexpected",
        "frame_missing",
    }
)


def _label(entity: str, name: str) -> str:
    return f"{entity}.{name}"


def _unreached_declaration(row: Mapping[str, object]) -> Mapping[str, object] | None:
    value = row.get("unreached_roots")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "sha256",
        "outputs",
        "names",
        "reason",
    }:
        raise ValueError(
            f"rules binding {row.get('id')!r}.unreached_roots holds exactly "
            "sha256, outputs, names and reason."
        )
    return value


def closure_partition(
    closure: InputClosure,
    mapping: ConceptMapping,
    bindings: Sequence[Mapping[str, object]],
    roots: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    module_sha256: Mapping[str, str],
    module_imports: Mapping[str, bool],
    mapped_entities: Iterable[str],
    frame_columns: Mapping[str, frozenset[str]] | None = None,
    entity_names: Mapping[str, str] | None = None,
    person_entity: str = "person",
) -> dict[str, object]:
    """Partition every live root of every bound module, and run the closure check.

    Args:
        closure: The reviewed closure.
        mapping: The country's concept mapping.
        bindings: ``axiom_rules_bindings.bindings`` rows, in file order.
        roots: Binding id -> engine entity -> the root input names the bound
            module reads on that entity.
        module_sha256: Binding id -> SHA-256 of the module bytes the run binds.
        module_imports: Binding id -> whether the module declares ``imports:``.
        mapped_entities: The engine entities the schema maps; roots on any
            other entity are reported as ``unmapped_entity`` and are not live.
        frame_columns: Frame entity -> columns the producer's slice carries,
            or ``None`` to skip the frame checks (engine-free tests).
        entity_names: Frame entity -> engine entity, needed with
            ``frame_columns``.
        person_entity: The frame's person entity, whose table carries every
            relation input's physical column.

    Returns:
        ``{"check": [...], "modules": [...]}``: ``check`` is
        :meth:`InputClosure.check` on the live surface (stamped with the
        closure's commits) and the mapping without unreached refs; each
        module row puts every live root in exactly one of
        :data:`CLOSURE_CLASSES`.

    Raises:
        ValueError: Naming every refused ``unreached_roots`` declaration, or a
            binding that declares a binding-local view.
    """

    mapped = frozenset(mapped_entities)
    to_frame = {engine: frame for frame, engine in (entity_names or {}).items()}
    refusals: list[str] = []
    excused: set[InputRef] = set()
    declared_unreached: dict[str, frozenset[str]] = {}
    live_roots: dict[str, dict[str, tuple[str, ...]]] = {}
    unmapped_roots: dict[str, list[str]] = {}
    knob_targets = {
        target
        for knob in closure.knobs
        if knob.kind is not KnobKind.STATE_COLUMN
        for target in knob.targets
    }
    for row in bindings:
        binding = str(row["id"])
        path = str(row["rulespec_path"])
        if "views" in row:
            refusals.append(
                BINDING_LOCAL_VIEW_MESSAGE.format(
                    module=path, entity=row.get("engine_entity"), input="*"
                )
            )
        by_entity = roots.get(binding, {})
        live_roots[binding] = {
            entity: tuple(sorted(set(names)))
            for entity, names in sorted(by_entity.items())
            if entity in mapped
        }
        unmapped_roots[binding] = sorted(
            _label(entity, name)
            for entity, names in by_entity.items()
            if entity not in mapped
            for name in names
        )
        declaration = _unreached_declaration(row)
        if declaration is None:
            declared_unreached[binding] = frozenset()
            continue
        label = f"rules binding {binding!r}.unreached_roots"
        names = declaration["names"]
        outputs = declaration["outputs"]
        reason = declaration["reason"]
        if not isinstance(reason, str) or not reason.strip():
            refusals.append(f"{label} needs a reason.")
        if declaration["sha256"] != row.get("sha256") or declaration[
            "sha256"
        ] != module_sha256.get(binding):
            refusals.append(
                f"{label}.sha256 {declaration['sha256']} is not the binding's "
                f"module SHA-256 {row.get('sha256')}; re-verify the unreached roots."
            )
        if not isinstance(outputs, list | tuple) or list(outputs) != list(
            row.get("variables", ())
        ):
            refusals.append(
                f"{label}.outputs {outputs!r} differ from the binding's variables "
                f"{list(row.get('variables', ()))!r}."
            )
        if module_imports.get(binding, False):
            refusals.append(
                f"{label}: module {path} declares imports, which its SHA-256 does "
                "not cover."
            )
        if (
            not isinstance(names, list | tuple)
            or not names
            or len(set(names)) != len(names)
            or not all(isinstance(name, str) and name for name in names)
        ):
            refusals.append(f"{label}.names must be distinct non-empty strings.")
            declared_unreached[binding] = frozenset()
            continue
        entity = str(row.get("engine_entity"))
        live = set(live_roots[binding].get(entity, ()))
        closed = {
            entry.input for entry in closure.entries_for(path) if entry.entity == entity
        }
        for name in names:
            if name not in live:
                refusals.append(f"{label} names {name!r}, which is not a live root.")
            if name in closed:
                refusals.append(
                    f"{label} names {name!r}, which the closure also closes."
                )
        declared_unreached[binding] = frozenset(names)
        excused.update(InputRef(name, entity, path) for name in names)
    if refusals:
        raise ValueError(
            "transport.input_closure@1 refuses the rules bindings: "
            + " ".join(refusals)
        )

    surface = AxiomInputSurface(
        country=closure.country,
        rulespec_commit=closure.surface_rulespec_commit,
        engine_commit=closure.surface_engine_commit,
        engine_surface="live",
        modules=MappingProxyType(
            {
                str(row["rulespec_path"]): AxiomModuleSurface(
                    path=str(row["rulespec_path"]),
                    sha256=module_sha256[str(row["id"])],
                    status="compiled",
                    inputs=MappingProxyType(live_roots[str(row["id"])]),
                    canonical_inputs=MappingProxyType({}),
                )
                for row in bindings
            }
        ),
    )
    checked_surface, checked_mapping = without_unreached(
        surface, mapping, excused, mapped
    )
    problems = list(closure.check(checked_mapping, checked_surface))

    modules = []
    for row in bindings:
        binding = str(row["id"])
        path = str(row["rulespec_path"])
        relations = {item.input: item for item in binding_relation_inputs(row)}
        entries = {entry.ref: entry for entry in closure.entries_for(path)}
        classes: dict[str, list] = {name: [] for name in CLOSURE_CLASSES}
        encoded: dict[str, list[str]] = {by.value: [] for by in EncodedBy}
        frame_missing: list[str] = []
        roots_out: list[str] = []
        listed = path in closure.modules
        for entity, names in live_roots[binding].items():
            frame_entity = to_frame.get(entity)
            carried = (
                None
                if frame_columns is None
                else frame_columns.get(frame_entity, frozenset())
            )
            for name in names:
                ref = InputRef(name, entity, path)
                label = _label(entity, name)
                roots_out.append(label)
                entry = entries.get(ref) if listed else None
                if name in declared_unreached[binding] and entity == row.get(
                    "engine_entity"
                ):
                    classes["unreached"].append(label)
                elif entry is not None and entry.closure_class is ClosureClass.ENCODED:
                    if name in knob_targets:
                        classes["knob_targeted"].append(label)
                    else:
                        encoded[entry.encoded_by.value].append(label)
                    if (
                        carried is not None
                        and entry.encoded_by is not EncodedBy.GRAPH_NODE
                        and name not in carried
                    ):
                        frame_missing.append(label)
                elif (
                    entry is not None and entry.closure_class is ClosureClass.DEFAULTED
                ):
                    classes["defaulted"].append(
                        {
                            "input": label,
                            "value": entry.value,
                            "citation": entry.citation,
                            "new_default": entry.new_default,
                        }
                    )
                    if carried is not None and name not in carried:
                        frame_missing.append(label)
                elif entry is not None:
                    classes["engine_optional"].append(label)
                elif name in relations:
                    relation = relations[name]
                    if frame_columns is not None and relation.column not in (
                        frame_columns.get(person_entity, frozenset())
                    ):
                        classes["gap"].append(label)
                    else:
                        classes["relation"].append(
                            {
                                "input": label,
                                "column": relation.column,
                                "role_column": relation.role_column,
                            }
                        )
                else:
                    classes["gap"].append(label)
        live = {
            InputRef(name, entity, path)
            for entity, names in live_roots[binding].items()
            for name in names
        }
        unexpected = (
            sorted(_label(ref.entity, ref.name) for ref in entries if ref not in live)
            if listed
            else []
        )
        modules.append(
            {
                "binding": binding,
                "rulespec_path": path,
                "sha256": module_sha256[binding],
                "reviewed_sha256": closure.modules.get(path),
                "engine_entity": row.get("engine_entity"),
                "roots": sorted(roots_out),
                "unmapped_entity": unmapped_roots[binding],
                "encoded": {
                    key: sorted(value) for key, value in sorted(encoded.items())
                },
                "knob_targeted": sorted(classes["knob_targeted"]),
                "defaulted": sorted(
                    classes["defaulted"], key=lambda item: item["input"]
                ),
                "engine_optional": sorted(classes["engine_optional"]),
                "relation": sorted(classes["relation"], key=lambda item: item["input"]),
                "unreached": sorted(classes["unreached"]),
                "gap": sorted(classes["gap"]),
                "unexpected": unexpected,
                "frame_missing": sorted(frame_missing),
            }
        )
    return {"check": problems, "modules": modules}


def partition_counts(row: Mapping[str, object]) -> dict[str, int]:
    """Live roots per class of one module row; they sum to ``len(roots)``."""

    return {
        "encoded": sum(len(names) for names in row["encoded"].values()),
        **{name: len(row[name]) for name in CLOSURE_CLASSES if name != "encoded"},
    }
