"""Explicit fiscal input policy, without source or complete-parent authority.

Documents describe reviewed intent, not verified approval. No assumptions are
enabled in a US candidate by this module. A dynamic trace of one population
cannot prove universal inactivity, so inactive entries are unsupported.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph.canonical import canonical_json

MAX_POLICY_BYTES = 1024**2
_POLICY_KIND = "microcosm.us.fiscal_leaf_policy"
_PRODUCER_FIELDS = {"kind", "entity", "producer"}
_ASSUMPTION_FIELDS = {
    "kind",
    "entity",
    "value",
    "interpretation",
    "period",
    "roots",
    "affected_roots",
    "affected_programs",
    "rationale",
    "reviewer",
    "source_issue",
    "reform_sensitivity",
}


def _require(condition, reason):
    if not condition:
        raise ValueError("FISCAL_LEAF_POLICY_" + reason)


def _text(value):
    return type(value) is str and bool(value.strip())


def _names(value):
    return (
        type(value) is list
        and bool(value)
        and all(_text(item) for item in value)
        and len(set(value)) == len(value)
    )


def _literal(value):
    return (
        type(value) is bool
        or (type(value) is int and -(2**63) <= value < 2**63)
        or (type(value) is float and math.isfinite(value))
    )


@dataclass(frozen=True)
class FiscalLeafPolicy:
    """Exact bytes and digest; descriptive only, never a parent-admission token."""

    payload: bytes
    sha256: str


def load_fiscal_leaf_policy(
    payload: bytes, *, expected_sha256: str
) -> FiscalLeafPolicy:
    """Verify caller-supplied policy bytes against an explicitly reviewed pin.

    No file or network access. An entry's reviewer/issue text is provenance,
    not authenticated approval and not permission to run assumptions.
    """
    result = FiscalLeafPolicy(payload, expected_sha256)
    fiscal_leaf_policy_document(result)
    return result


def fiscal_leaf_policy_document(policy: FiscalLeafPolicy) -> dict:
    """Recheck the pin and schema, returning a detached parsed document."""
    _require(type(policy) is FiscalLeafPolicy, "TYPE")
    _require(
        type(policy.payload) is bytes and 0 < len(policy.payload) <= MAX_POLICY_BYTES,
        "BOUND",
    )
    _require(
        type(policy.sha256) is str
        and hashlib.sha256(policy.payload).hexdigest() == policy.sha256,
        "FINGERPRINT",
    )
    try:
        raw = json.loads(policy.payload)
        canonical = canonical_json(raw)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("FISCAL_LEAF_POLICY_JSON") from None
    _require(canonical == policy.payload, "CANONICAL")
    _require(
        type(raw) is dict
        and set(raw)
        == {"schema_version", "artifact_kind", "period", "roots", "entries"},
        "FIELDS",
    )
    _require(
        type(raw["schema_version"]) is int
        and raw["schema_version"] == 1
        and raw["artifact_kind"] == _POLICY_KIND,
        "SCHEMA",
    )
    _require(
        type(raw["period"]) is int and raw["period"] > 0 and _names(raw["roots"]),
        "SCOPE",
    )
    _require(
        type(raw["entries"]) is dict and all(_text(name) for name in raw["entries"]),
        "ENTRIES",
    )
    for entry in raw["entries"].values():
        _require(type(entry) is dict, "ENTRY")
        kind = entry.get("kind")
        _require(kind != "inactive", "INACTIVE_UNSUPPORTED")
        _require(kind in ("producer", "assumption"), "ENTRY_KIND")
        _require(entry.get("entity") in US_SCHEMA.entities, "ENTITY")
        if kind == "producer":
            _require(
                set(entry) == _PRODUCER_FIELDS and _text(entry["producer"]),
                "PRODUCER_FIELDS",
            )
            continue
        _require(set(entry) == _ASSUMPTION_FIELDS, "ASSUMPTION_FIELDS")
        _require(_literal(entry["value"]), "LITERAL_UNSUPPORTED")
        _require(
            entry["interpretation"] in ("baseline_behavior", "scenario_parameter"),
            "INTERPRETATION",
        )
        _require(
            type(entry["period"]) is int
            and entry["period"] == raw["period"]
            and entry["roots"] == raw["roots"],
            "ASSUMPTION_SCOPE",
        )
        _require(
            _names(entry["affected_roots"])
            and set(entry["affected_roots"]) <= set(raw["roots"])
            and _names(entry["affected_programs"]),
            "AFFECTED_SCOPE",
        )
        _require(
            all(
                _text(entry[key])
                for key in (
                    "rationale",
                    "reviewer",
                    "source_issue",
                    "reform_sensitivity",
                )
            ),
            "REVIEW_SCOPE",
        )
    return raw


def classify_fiscal_leaf_policy(policy, *, period, roots, leaves):
    """Bind a policy exactly to the independently derived static input closure."""
    raw = fiscal_leaf_policy_document(policy)
    _require(raw["period"] == period, "POLICY_PERIOD")
    _require(raw["roots"] == list(roots), "POLICY_ROOTS")
    _require(set(leaves) <= set(raw["entries"]), "UNPOLICIED_MODEL_LEAF")
    _require(set(raw["entries"]) <= set(leaves), "STALE_MODEL_LEAF")
    _require(
        all(entry["entity"] == leaves[name] for name, entry in raw["entries"].items()),
        "LEAF_ENTITY",
    )
    return raw


def fiscal_assumption_engine_records(document, *, metadata, defaults):
    """Validate literal types and retain actual engine metadata/default records.

    The caller derives metadata/defaults from the engine pinned in the fiscal
    contract. Defaults are recorded, never selected by reference. Enum/string
    assumptions require an additional reviewed representation and are refused.
    """
    records = {}
    for name, entry in document["entries"].items():
        if entry["kind"] != "assumption":
            continue
        item = metadata[name]
        _require(
            item["name"] == name and item["entity"] == entry["entity"], "ENGINE_ENTITY"
        )
        value = entry["value"]
        _require(
            (item["dtype"] == "bool" and type(value) is bool)
            or (item["dtype"] == "int" and type(value) is int)
            or (item["dtype"] == "float" and type(value) in (int, float)),
            "ENGINE_LITERAL_TYPE",
        )
        default = {"available": name in defaults}
        if name in defaults:
            _require(_literal(defaults[name]), "ENGINE_DEFAULT_UNSUPPORTED")
            default["value"] = defaults[name]
        records[name] = {"metadata": dict(item), "default": default}
    return records


def validate_fiscal_leaf_policy_population(
    frame: Frame, policy: FiscalLeafPolicy
) -> None:
    """Refuse any incumbent assumption column in an actual complete Frame.

    The country host must supply its independently admitted complete parent,
    before projection and again around cache/export I/O. Passing a projection
    is not absence proof. This function does not issue or authenticate a Frame.
    Even wholly or partially unknown incumbent columns count as collisions.
    """
    raw = fiscal_leaf_policy_document(policy)
    _require(frame.schema == US_SCHEMA, "POPULATION_SCHEMA")
    supplied = {
        column for entity in frame.entities for column in frame.table(entity).columns
    }
    for name, entry in raw["entries"].items():
        if entry["kind"] == "assumption":
            _require(name not in supplied, "ASSUMPTION_COLUMN_COLLISION:" + name)


def private_fiscal_assumption_frame(frame: Frame, policy: FiscalLeafPolicy) -> Frame:
    """Prepare detached engine-only tables; not called by the fiscal kernel.

    Future host integration must separately admit the complete parent, bind
    policy plus engine records, and call the collision validator around I/O.
    This pure helper is not an assumption-enabled graph execution path.
    """
    validate_fiscal_leaf_policy_population(frame, policy)
    tables = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    tables.update({name: frame.link(name).copy(deep=True) for name in frame.links})
    for name, entry in fiscal_leaf_policy_document(policy)["entries"].items():
        if entry["kind"] == "assumption":
            tables[entry["entity"]][name] = np.full(
                frame.n(entry["entity"]), entry["value"]
            )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        strata=frame.strata.copy(deep=True),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
